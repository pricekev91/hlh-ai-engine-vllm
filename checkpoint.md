# Checkpoint — vLLM project wheels (wheels.vllm.ai), resume point 2026-09-19

> Replaces the venv-era checkpoint (that content is preserved in git history,
> e.g. `git show 99a1757:checkpoint.md`). Everything below is verified live
> state unless marked "plan".

## 1. Where we are (TL;DR)

- **Live LXC 113 is healthy on the 0.4.0 stack** (AMD prebuilt wheels from
  `rocm.frameworks.amd.com`: vllm 0.27.1.dev5+rocm10.0.0, torch 2.12.0+rocm10.0.0,
  Python 3.14.7). It currently serves **`Qwen3.6-35B-A3B-GPTQ-Int4`**
  (user-switched, not the README default), health 200.
- **Decision: switch the configure stack to the vLLM project's own ROCm wheels**
  (`https://wheels.vllm.ai/rocm/<ver>/<rocmvariant>/`) — the path documented on
  docs.vllm.ai. For vLLM 0.29.0 the project ships **only** `rocm723` (ROCm 7.2.3).
- **Decision: do NOT hard-pin 0.29.0/rocm723.** The configure script must
  *resolve at deploy time*: latest **stable** vLLM (PyPI) → matching ROCm
  variant (newest `rocm*` dir published for that version). Today that resolves
  to 0.29.0/rocm723; when 0.30.0 ships with a rocm10.x dir, it picks that up
  with zero script changes. Env overrides: `VLLM_VERSION`, `VLLM_ROCM_VARIANT`.
- **Not done yet:** script rewrite + live test. Live LXC is untouched.

## 2. Verified facts (2026-09-19)

### Proxmox host 192.168.1.10 (prox01)
- **No `amdrocm*` apt packages installed.** `/usr/bin/rocm-smi` is an orphan
  (ROCM-SMI 4.0.0 / ROCM-SMI-LIB 7.8.0, `dpkg -S` owns nothing);
  `rocm.list` was renamed `rocm.list.bak`; `/opt/rocm` has `core-7.14` +
  `core-10.0` leftovers only.
- Kernel **7.0.14-11-pve** with built-in amdgpu driver is the *only* host
  component the LXC needs (driver + firmware behind `/dev/kfd`). It demonstrably
  works: 113 serves GPU workloads today.
- LXC **112 is running** (co-tenant, same 890M iGPU — never stop it).
- **Conclusion:** the deploy script's "host ROCm 10.0.0 apt upgrade" logic is
  not required for a wheel-based stack (the LXC userspace comes entirely from
  pip wheels). Host check should become **informational / opt-in**
  (`HOST_ROCM_SETUP=1`), not a forced upgrade prompt. (Also: the current
  `get_host_rocm_version` fallback reads the orphan rocm-smi → reports
  "4.0.0" and would prompt a bogus "upgrade" on every run.)

### LXC 192.168.1.13 (113)
- `vllm.service` active; `/etc/vllm.env` uses `AI_*` vars; model
  `/srv/ai/models/Qwen3.6-35B-A3B-GPTQ-Int4` served as
  `qwen3.6-35b-a3b-gptq-int4`; `AI_GPU_MEM_UTIL=0.40`, `AI_MAX_MODEL_LEN=4096`.
- venv (`/opt/vllm-venv`, Python 3.14.7): torch 2.12.0+rocm10.0.0,
  torchvision 0.27.0+rocm10.0.0, torchaudio 2.11.0+rocm10.0.0,
  flash_attn 2.8.3, amd-aiter 0.1.20.post1,
  vllm 0.27.1.dev5+rocm10.0.0.gf46a9dfe2.d20260826.
- `curl :8000/health` → 200.

### wheels.vllm.ai (project index)
- `rocm/0.29.0/` → exists (HTTP 200), **only variant: `rocm723/`**.
  (`rocm/0.30.0/` → 404, `rocm/0.29.1/` → 404, `rocm/0.28.0/` → 200.)
- `rocm723/` is a **self-contained simple index** with subdirs:
  `torch/ torchvision/ torchaudio/ triton/ flash-attn/ amd-aiter/ amdsmi/ vllm/`
  — one `--index-url` covers the whole stack.
- `vllm/` contents:
  - `vllm-0.29.0+rocm723-cp312-cp312-manylinux_2_39_x86_64.whl` ← **stable**
  - `vllm-0.29.0rc7.dev1+g98dff2a81.rocm723-cp312-...whl` (pre-release)
  - `vllm-0.29.1.dev1+g98dff2a81.rocm723-cp312-...whl` (pre-release)
- `torch/`: `torch-2.12.0+git6bbd260-cp312-cp312-manylinux_2_39_x86_64.whl`
  (same 2.12.0 series as AMD's `+rocm10.0.0`, different build).
- `flash-attn/`: 2.8.3 cp312; `amd-aiter/`: 0.1.19 cp312.
- **All wheels are cp312** → Python 3.12 (live LXC is 3.14.7 — venv must be
  rebuilt for 3.12). **manylinux_2_39** → needs glibc ≥ 2.39 (Ubuntu 24.04 =
  2.39 ✓).
- **The S3 root `wheels.vllm.ai/rocm/` is not listable** (no dir index) →
  "latest stable" discovery must go via **PyPI**:
  `https://pypi.org/pypi/vllm/json` → `info.version` (currently **0.29.0**,
  matching the newest existing rocm dir).

## 3. Resolution design (plan for new configure script, v0.5.0)

1. **vLLM version:** `VLLM_VERSION` env → pin. Else: PyPI JSON `info.version`;
   verify `https://wheels.vllm.ai/rocm/<ver>/` → 200; on 404 walk PyPI
   `releases` newest→oldest (skip rc/post/dev) until a 200 is found. Fail loud
   if none.
2. **ROCm variant:** `VLLM_ROCM_VARIANT` env → pin (e.g. `723`). Else list
   `rocm[0-9]+/` dirs under the version dir → pick **highest**
   (parse `NNN`/`NNNN` as major + 1-digit minor + 1-digit patch: 723→7.2.3,
   1000→10.0.0). Fail loud if none.
3. **Python version:** GET `.../<ver>/<variant>/vllm/`, pick the **stable**
   wheel (`grep -vE 'rc|\.dev|dev[0-9]'`), parse `cp3NN` tag →
   `uv venv --python 3.NN` (uv fetches standalone CPython as needed).
4. **Install (uv) — index order matters:**
   ```
   uv pip install --python "$PY" \
     --index-url  "https://wheels.vllm.ai/rocm/$V/$R/" \
     --extra-index-url "https://pypi.org/simple" \
     vllm
   ```
   PyPI's `vllm` is the **CUDA** build. uv's default first-match resolution
   takes a package from the *first* index that has it → the ROCm index must be
   primary so `vllm`/`torch`/`triton`/`amdsmi` resolve to the `+rocm723`
   builds, while everything else (transformers, etc.) falls through to PyPI.
   (The docs' `--extra-index-url` one-liner is pip-flavored; with pip the
   `+rocm723` local version also wins, but with uv the index must be first —
   verify with `uv pip install --dry-run` before the live run.)
   Then optionally `flash-attn amd-aiter` from the same index — warn-and-continue
   if unavailable (Triton FA fallback via `FLASH_ATTENTION_TRITON_AMD_ENABLE`).
5. **Hard verification (keep, fail loud):** no `nvidia-`/`cuda-` packages in
   freeze; `torch.version.hip` present; `torch.cuda.is_available()` +
   `mem_get_info`; `import vllm`; print resolved versions.
6. **Runtime files (keep as-is):** `/etc/vllm.env` (`AI_*` vars),
   `/usr/local/bin/vllm-run.sh` (`vllm serve`, `--enforce-eager`,
   `--tool-call-parser qwen3_coder`, MM limits, `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`),
   `vllm.service` (StartLimit backoff), `vllm-switch-model.sh`.
   **Remove:** `AMD_SMI_PATH` PYTHONPATH hack (project wheel ships `amdsmi`
   properly via the index), hardcoded `TORCH_SPEC`/`TORCHVISION`/`TORCHAUDIO`,
   `stable.repo.amd.com/rocm/whl-next` and `rocm.frameworks.amd.com` URLs.
   **HSA override:** NOT set by default; keep in `/etc/vllm.env` as a commented
   last-resort `HSA_OVERRIDE_GFX_VERSION=11.0.0`.

## 4. Deploy script changes (plan)

- SBOM header → "vLLM **latest stable** + matching ROCm variant, auto-resolved
  from wheels.vllm.ai at deploy time (today: 0.29.0/rocm723)".
- `ROCM_VERSION` (default 10.0.0) + forced host apt-upgrade prompt →
  **opt-in** (`HOST_ROCM_SETUP=1`); default = informational (report kernel,
  `/dev/kfd`, host rocm-smi if present).
- Forward `VLLM_VERSION` / `VLLM_ROCM_VARIANT` into the LXC bootstrap;
  back-compat: if user passes `ROCM_VERSION=x.y.z`, derive variant
  (`7.2.3`→`723`, `10.0.0`→`1000`) and forward as `VLLM_ROCM_VARIANT`.

## 5. Risks / open questions

- **gfx1150 kernel coverage in the project's rocm723 torch build is
  unverified** — this is the main "I'll make it work" item. Fallback:
  `HSA_OVERRIDE_GFX_VERSION=11.0.0` (proven with the CUDA-wheel era stack).
  If the rocm723 torch wheel lacks gfx1150/gfx1151 kernels, first kernel
  launch will error (`invalid device function` class) → set the override.
- uv first-match behavior with dual indexes — confirm via `--dry-run`
  (expect `vllm 0.29.0+rocm723` + `torch 2.12.0+git6bbd260`, not PyPI CUDA).
- After the stack swap, the **35B-A3B GPTQ** model (currently served) must
  still load and answer; re-check co-tenancy memory (`free -g` / host
  `rocm-smi --showmeminfo vram`) with 112 running.
- **Doc debt (follow-up):** README/`10_ACTIVE.md`/`90_DONE.md` still describe
  the 0.2.0 docker / 0.3.x CUDA-wheel era (HSA "mandatory",
  `VLLM_USE_V2_MODEL_RUNNER=0` "mandatory", `VLLM_*` env names,
  "torch 2.8.0+rocm6.4", deploy SBOM "vLLM 0.29.0 — git pull" line).
  The live truth is 0.4.0 today and 0.5.0 after this change.

## 6. Next steps (in order)

1. Rewrite `configure-hlh-ai-engine-vllm.sh` (v0.5.0) per §3.
2. Edit `deploy-hlh-ai-engine-vllm.sh` per §4; `bash -n` + shellcheck both.
3. CHANGELOG `[0.5.0]` entry; focused README fixes (SBOM line, backend notes,
   overrides, env table); `10_ACTIVE.md` note.
4. Live test on the host: `./deploy-hlh-ai-engine-vllm.sh --update`
   (in-place venv rebuild; models persist on ZFS; ~5–15 min). Watch
   `journalctl -u vllm -f`; first-kernel-launch errors → add HSA override to
   `/etc/vllm.env`, restart, iterate.
5. Verify: `/health` 200, `/v1/models`, chat completion on
   `qwen3.6-35b-a3b-gptq-int4`, memory under co-tenancy.
6. Commit + push.

## 7. Flash-attention clarification (user Q, 2026-09-19)

- **llama.cpp (LXC 112):** flash attention is compiled *into the ggml binary*
  (HIP kernels). Separate thing, no packages.
- **vLLM (LXC 113):** `flash_attn` / `amd-aiter` pip packages are the
  **attention compute kernel only** (Q×K×V matmul+softmax). They do **not**
  change vLLM's memory model: memory is `--gpu-memory-utilization` KV-pool
  pre-allocation + PagedAttention blocks, independent of the attention kernel.
  On the 890M APU the kernel choice affects *speed/availability* (hence
  `--enforce-eager` + `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` Triton
  fallback), never the memory accounting. Don't conflate FA presence with
  `AI_GPU_MEM_UTIL` tuning.

## 8. 0.5.1 — torch import failure (2026-09-20)

- **Symptom:** fresh LXC 113, `deploy --update`-style bootstrap died at [4/8]:
  `ImportError: libroctx64.so.4: cannot open shared object file` (plus 14 more
  missing ROCm libs in `ldd torch/lib/*.so`).
- **Root cause (verified live):** LXC template has **no `gnupg`** →
  `wget rocm.gpg.key | gpg --dearmor | tee amdrocm.gpg` wrote a **0-byte keyring**
  (all stages masked by `>/dev/null` + `|| true`) → `apt-get update` failed
  `GPG error: NO_PUBKEY 9386B48A1A693C5C` (swallowed) → all three
  `apt-get install rocm…` fallbacks: `Unable to locate package` → warn-and-continue
  → torch crash. Keyring at `/etc/apt/keyrings/amdrocm.gpg` was 0 bytes on the
  failed box; `command -v gpg` → nothing.
- **Fix (0.5.1):** `gnupg` in base packages + hard `gpg` gate; `fetch_rocm_key()`
  with retry + non-empty + `gpg --show-keys` verification; repo URL probed
  (dotted 7.2.3 → 7.2 → 10.x dists); `apt-get update` logged + `rocm-core`
  candidate hard gate; fallback lib list corrected (drop nonexistent
  `hiprtc-amd`, add `hsa-amd-aqlprofile` + `libdw1`); early `ldconfig -p` gate
  on 4 core libs; [4/8] missing-libs now FATAL; `ROCM_LIB_DIRS`/`_ROCM_HOME`
  derived from real `/opt/rocm*` (was hardcoded `/opt/rocm-7.2.0`).
- **Verified live (ad-hoc first, then full `deploy --update` re-run):** keyring
  2250 bytes, repo `https://repo.radeon.com/rocm/apt/7.2.3 noble main`, libs in
  `/opt/rocm-7.2.3`, torch 2.12.0+git6bbd260 hip 7.2.53211, GPU "AMD Radeon 890M",
  cuda matmul OK, vllm 0.29.0 imports.

## 9. Quick resume commands

```bash
# local
cd ~/git/hlh-ai-engine-vllm && git log --oneline -5

# live state
ssh root@192.168.1.13 'systemctl is-active vllm; curl -s -m3 http://127.0.0.1:8000/health -o /dev/null -w "%{http_code}\n"; grep -E "^(AI_MODEL_PATH|AI_SERVED_NAME)=" /etc/vllm.env; /opt/vllm-venv/bin/pip list 2>/dev/null | grep -iE "^(vllm|torch) "'
ssh root@192.168.1.10 'pct status 112; pct status 113'

# index probes
curl -sL https://pypi.org/pypi/vllm/json | python3 -c "import sys,json;print(json.load(sys.stdin)['info']['version'])"
curl -sL "https://wheels.vllm.ai/rocm/0.29.0/" | grep -oE 'rocm[0-9]+/'
curl -sL "https://wheels.vllm.ai/rocm/0.29.0/rocm723/vllm/" | grep -oE '>[^<]+\.whl' | tr -d '>'
```
