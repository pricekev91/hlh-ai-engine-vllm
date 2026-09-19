# Checkpoint — vLLM on LXC 113 (2026-09-16, late evening)

> Resume point for getting the 7B GPTQ model serving on `root@192.168.1.13`.
> Written before going to bed; everything below is verified live state, not aspiration.

## 1. Where we are (TL;DR)

- vLLM 0.29.0 (PyPI = CUDA-built wheel, **no ROCm C extensions**) on torch 2.8.0+rocm6.4, Radeon 890M (gfx1150), 48 GB shared RAM.
- All 7 earlier startup blockers are fixed (see §5). Service is **stopped** (was crash-looping) to keep the box quiet overnight.
- **Current single blocker:** GPTQ zero-points (`qzeros`) are not unpacked before the RDNA W4A16 Triton kernel:
  ```
  AssertionError: zp shape mismatch: torch.Size([28, 576]) vs (4608, 28)
  ```
  Stack: `auto_gptq.py:464 apply` → `kernels/linear/mixed_precision/rdna_hybrid_w4a16.py:551 apply_weights` → `:408 _rdna_hybrid_w4a16_apply_impl` → `:228 triton_w4a16_skinny_fmt_gemm` (assert `zp.shape == (N, num_groups)`).
  - Failing layer = **fused qkv_proj** of layer 0: N = 3584(q)+512(k)+512(v) = **4608**, K = 3584, num_groups = 28 (group_size 128).
  - Actual zp `(28, 576)` is the **raw HF packed qzeros layout** `(K/group, N/8)` (int32-packed 4-bit zps). It was never unpacked/transposed to `(N, K/group)`.
  - `scales` unpacked fine (its assert passed) — only `zp` stays packed.
  - Model checkpoint **does contain qzeros** (196 `*.qzeros` tensors in `model.safetensors.index.json`) even though `quantization_config.sym = true`. `desc_act = false` (no act-order shuffle needed — good, that shuffle is a missing C op we don't have).
- vLLM 0.29 ships a pure-Triton RDNA W4A16 GPTQ kernel (`rdna_hybrid_w4a16`) built for exactly this hardware — no C extension needed. We are one loader bug away, not one missing wheel away.

## 2. Environment facts

| Thing | Value |
|---|---|
| Proxmox host | `root@192.168.1.10` (prox01) |
| LXC | `root@192.168.1.13` = LXC 113 `hlh-ai-engine-vllm`, Ubuntu 24.04, privileged |
| HW | AMD Ryzen AI 9 HX 370 (Strix Halo), Radeon 890M = gfx1150, 48 GB RAM (unified) |
| Local repo | `/home/pricekev/git/hlh-ai-engine-vllm` |
| vLLM | 0.29.0, plain PyPI wheel (CUDA build: `vllm._C_stable_libtorch.abi3.so` needs `libcudart.so.13`; **no** `vllm._C`, **no** `vllm._rocm_C`) |
| torch | 2.8.0+rocm6.4 (HIP). venv: `/opt/vllm-venv` |
| Other kernels pkgs | triton 3.7.1, pytorch-triton-rocm 3.4.0, humming-kernels, quack-kernels, tokenspeed-triton. **No aiter.** |
| Services | `vllm.service` (:8000), Open WebUI (:80) |
| Models dir | LXC `/srv/ai/models` = host `/srv/ai/models` (ZFS bind, persists across nuke) |

Models on disk:
- `Qwen2.5-Coder-7B-Instruct-GPTQ-Int4` — Qwen2ForCausalLM, hidden 3584, inter 18944, heads 28×128, KV 4×128. GPTQ 4-bit, group 128, `desc_act=false`, `sym=true` (but qzeros present).
- `Qwen3.6-27B-GPTQ-Int4` — `Qwen3_5ForConditionalGeneration`, `qwen3_5`, gptq.
- `Qwen3.6-35B-A3B-GPTQ-Int4` — `Qwen3_5MoeForConditionalGeneration`, `qwen3_5_moe`, gptq.

No official vLLM ROCm wheel is available in this environment (checked):
`https://wheels.vllm.ai/` → NoSuchKey; `pip index versions vllm --index-url https://wheels.vllm.ai` → nothing; `download.pytorch.org/whl/rocm6.4` has no vllm. So "install the ROCm build + aiter" (the textbook path) is not on the table; CUDA wheel + shims/patches is the viable route, and vLLM's native Triton RDNA kernels make it workable.

## 3. Live state on 192.168.1.13 (manual, NOT yet in IaC)

### 3.1 `/etc/systemd/system/vllm.service`
```
Environment=HSA_OVERRIDE_GFX_VERSION=11.0.0        # was 11.5.0 (broken, see §5.5)
Environment=VLLM_USE_V2_MODEL_RUNNER=0             # added (V2 runner needs missing C UVA op)
ExecStart: /opt/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /srv/ai/models/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4   # was /srv/ai/models/Qwen3.5-9B-safetensors (nonexistent)
  --host 0.0.0.0 --port 8000 \
  --served-model-name qwen2.5-coder-7b-instruct-gptq           # was qwen3.5-9b
  --gpu-memory-utilization 0.25                                # was 0.70 (OOM risk, see §7)
  --max-model-len 4096 --enforce-eager --dtype auto --trust-remote-code \
  --limit-mm-per-prompt '{"image":1,"video":1}' --mm-processor-cache-gb 1 \
  --skip-mm-profiling --max-num-seqs 8
```
Service currently **inactive** (stopped on purpose). `systemctl daemon-reload` was run after edits.

### 3.2 venv patches (all under `/opt/vllm-venv/lib/python3.12/site-packages/`)
1. **`vllm_rocm_accel_shim.py` + `vllm_rocm_accel_shim.pth`** (new files, site dir root):
   `.pth` contains `import vllm_rocm_accel_shim`; the shim (guarded, never breaks startup) backfills on ROCm builds:
   `torch.accelerator.{empty_cache, memory_stats, memory_allocated, max_memory_allocated, memory_reserved, reset_peak_memory_stats, get_memory_info→mem_get_info, empty_host_cache→no-op}` from `torch.cuda.*` (torch 2.8.0+rocm6.4 lacks them; vLLM 0.29 calls them).
2. **`vllm/v1/worker/gpu_worker.py`**: `torch.accelerator.empty_cache()` → `torch.cuda.empty_cache()` (pre-shim fix; now redundant, harmless).
3. **`vllm/model_executor/layers/activation.py`** (`SiluAndMul.__init__`): wrapped `self.op = torch.ops._C.silu_and_mul` in try/except → on `AttributeError/RuntimeError` sets `self._forward_method = self.forward_native`. (Mirrors the ROCm guard `SiluAndMulWithClamp` already had.)
4. **`vllm/kernels/vllm_c.py`** (module top): added `C_EXT_AVAILABLE = _c_ext_available()` (probes `torch.ops._C.rms_norm`) and changed `GPGPU_DEVICE = (CUDA_ALIKE or current_platform.is_xpu()) and C_EXT_AVAILABLE`. Effect: `vllm_c` provider for IR ops `rms_norm` / `fused_add_rms_norm` registers `supported=False`; IR dispatch (`_filter_priority_impls` in `vllm/ir/op.py`) drops it and falls through to `native`. Verified: `GPGPU_DEVICE=False`, `impls['vllm_c'].supported=False`.

### 3.3 Known-good facts
- `rocm-smi` works; reports 512 MB VRAM carve-out, ~178–200 MB used at idle.
- torch sees ~68.7 GB "GPU" total (GTT over the 48 GB system RAM) → `--gpu-memory-utilization 0.70` would ask ~48 GB and OOM the box. 0.25 ≈ 17 GB is safe.
- `rocm-smi --showmeminfo vram` + `free -g` are the two health gauges.

## 4. The debug chain (each item fixed in order)

1. Unit pointed at nonexistent `/srv/ai/models/Qwen3.5-9B-safetensors` (transformers misparsed it as a repo id) → repointed to 7B GPTQ dir.
2. `torch.accelerator.empty_cache` AttributeError → shim (§3.2.1).
3. `torch.accelerator.memory_stats` AttributeError → shim.
4. V2 model runner: `torch.ops._C.get_cuda_view_from_cpu_tensor` (UVA C op) missing → `VLLM_USE_V2_MODEL_RUNNER=0` (classic V1 runner; `config/vllm.py` documents the env).
5. V1 runner first kernel launch: `HIP error: invalid device function` → **HSA override must be 11.0.0**, not 11.5.0 (precompiled gfx1100 kernels; 11.5.0/gfx1150 native breaks kernel selection). Repo changelog 0.1.2 already noted "11.5.0 gives HIP invalid device"; 0.1.6 regressed it.
6. `SiluAndMul`: `torch.ops._C.silu_and_mul` missing at model construction → activation.py patch (§3.2.3).
7. `rms_norm` via `vllm_c` IR provider missing → vllm_c.py gating patch (§3.2.4).
8. **OPEN**: `zp shape mismatch: [28, 576] vs (4608, 28)` in `triton_w4a16_skinny_fmt_gemm` — qzeros not unpacked (see §1).

## 5. Next steps (do in this order)

All paths below on 192.168.1.13; `SP=/opt/vllm-venv/lib/python3.12/site-packages/vllm`.

1. Confirm raw qzeros layout of the failing layer:
   ```bash
   python3 - <<'EOF'
   import json, struct
   idx = json.load(open('/srv/ai/models/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4/model.safetensors.index.json'))
   key = 'model.layers.0.self_attn.q_proj.qzeros'   # qkv may be split q/k/v in the ckpt
   print(idx['weight_map'][key])
   # read the safetensors header of that shard, print shape/dtype of *.qzeros and *.scales
   EOF
   ```
   Expect qzeros `(28, 576)` int32, scales `(28, 4096)`? (verify; scales assert already passed at runtime as `(4608, 28)` after fusion).
2. Read `SP/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py` fully — especially `create_weights`, any `process_weights_after_loading`, and what zp dtype/shape `triton_w4a16_skinny_fmt_gemm` expects (assert at line ~228: `zp.shape == (N, num_groups)`, contiguous, probably float).
3. Read `SP/model_executor/layers/quantization/auto_gptq.py` (863 lines) — find where `qzeros` are converted to `zp` for MP linear kernels (search `qzeros`, `zp`, `process_`), and compare with how another kernel path unpacks qzeros (see also `SP/model_executor/layers/quantization/utils/` for existing unpack helpers, e.g. something like `process_gptq_qzeros`).
   Hypothesis to confirm: the unpack step is only wired for certain backends/branches, or is skipped because `sym=true` makes the loader think zps are absent — but the checkpoint ships qzeros anyway, and this kernel wants them unpacked.
4. Patch the loader so the rdna_hybrid kernel gets `zp` as `(N, K/group)` (int32 → 8×uint4 per element → float, transpose). Prefer reusing an existing unpack helper; keep it kernel-layout-conditional so other backends are untouched.
5. `systemctl restart vllm` (load cycle ≈ 2–4 min from ZFS); watch:
   ```bash
   journalctl -u vllm -f --no-pager        # EngineCore root cause = last 'Error' in core.py:1374 traceback
   curl -s http://127.0.0.1:8000/health
   rocm-smi --showmeminfo vram; free -g
   ```
6. When healthy, verify end-to-end:
   ```bash
   curl -s http://127.0.0.1:8000/v1/models
   curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model":"qwen2.5-coder-7b-instruct-gptq","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":16}'
   ```
7. Then proceed to the rest of the plan (§6). If the unpack patch is more than a few iterations of whack-a-mole, stop and write it up as a roadblock in this file.

## 6. Remaining work after 7B serves

1. **27B**: switch unit `--model /srv/ai/models/Qwen3.6-27B-GPTQ-Int4` + served name, raise `--gpu-memory-utilization` carefully (0.35–0.5; watch `free -g`). `qwen3_5` arch, dense, GPTQ → same kernel path; may hit `silu_and_mul_with_clamp` (already ROCm-guarded) or new C-op gaps.
2. **35B-A3B (MoE)**: `qwen3_5_moe` GPTQ → FusedMoE WNA16 backends (triton). Verify no `torch.ops._C` deps in the selected MoE path (`grep -rn "torch.ops._C" $SP/model_executor/layers/fused_moe* ...`), memory is the main risk (~18–20 GB weights + KV).
3. **Model switch script** `/srv/ai/models/vllm-switch-model.sh`: accept local dirs (the 3 models), stop vllm, edit/override unit (or env-based model path), restart, health-check. Must not clobber the patches above.
4. **Propagate into IaC (this repo)** — the nuke/deploy must reproduce the working box:
   - `configure-ai-engine-inside-lxc.sh`: correct default model, unit flags (HSA 11.0.0, `VLLM_USE_V2_MODEL_RUNNER=0`, `--gpu-memory-utilization 0.25`), and make the 4 venv shims/patches idempotent (drop shim+.pth files, run the same python patch scripts).
   - `deploy-hlh-ai-engine-vllm.sh`: default model + HSA override (LXC_ID=113, LXC_NAME=hlh-ai-engine-vllm, IP 192.168.1.13/24, POOL RaidZ1-6TB, models bind host `/srv/ai/models` → LXC `/srv/ai/models`). Pure bash.
   - `README.md`: kill all stale `Qwen3.5-9B-safetensors`/`qwen3.5-9b` references; document gfx1150 + HSA 11.0.0, the CUDA-wheel-without-C-extensions caveat + shims, per-model launch matrix and memory settings.
5. **Commit + push repo** (after 7B works, or at a clear roadblock) — user then tests and does the nuke/deploy themselves by running `deploy-hlh-ai-engine-vllm.sh` on 192.168.1.10 (destroys/recreates LXC 113 from scratch; models persist via the ZFS bind).
6. Out of scope (user said ignore): `nvidia-smi`/K80 "Unable to determine the device handle" on 192.168.1.10 (future V100/MI50 shopping).

## 7. Gotchas / lessons

- **`HSA_OVERRIDE_GFX_VERSION=11.0.0` is mandatory** for this vLLM/torch combo on gfx1150. 11.5.0 → `HIP error: invalid device function` at first kernel launch. (rocm-smi is unaffected either way.)
- **Memory**: `--gpu-memory-utilization` × ~68.7 GB GTT pool. 0.70 ≈ 48 GB ≈ whole system RAM → OOM. Use 0.25 for 7B; go up slowly for 27B/35B and watch `free -g` / `rocm-smi`.
- PyPI `vllm` is CUDA-only: anything in vLLM that hard-requires `torch.ops._C` must get a python fallback (pattern: try/except + native impl, as done for `SiluAndMul`, or provider gating as done for `vllm_c`).
- EngineCore errors: `journalctl -u vllm | grep "core.py:1374"`; the **last** `Error` line in that traceback is the root cause; APIServer traceback is the symptom.
- Each service restart cycle ≈ 2–4 min (5.5 GB weights off ZFS).
- `sym=true` in quantize_config does **not** mean qzeros are absent from the checkpoint — this one ships them.

## 8. Quick resume commands

```bash
ssh root@192.168.1.13
systemctl status vllm                      # expect: inactive (stopped overnight)
journalctl -u vllm -n 40 --no-pager        # last failure = zp shape mismatch
SP=/opt/vllm-venv/lib/python3.12/site-packages/vllm
sed -n '200,260p' $SP/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py
grep -n 'qzeros\|def process\|zp' $SP/model_executor/layers/quantization/auto_gptq.py | head -40
```
