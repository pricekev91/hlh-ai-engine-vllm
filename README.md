# hlh-ai-engine-vllm

Native vLLM OpenAI-compatible server (agentic coding) on a **Tesla V100 32GB eGPU (OCuLink, CUDA 12.8)**, plus a native Open WebUI chat UI — no Docker. Serves models from a shared ZFS pool.

## Quick start

```bash
cd ~/git/hlh-ai-engine-vllm
./deploy-hlh-ai-engine-vllm.sh                 # full deploy (creates LXC 113 + GPU passthrough + vLLM + WebUI)
./deploy-hlh-ai-engine-vllm.sh --skip-host-driver   # skip host driver re-validation
./deploy-hlh-ai-engine-vllm.sh --update       # fast in-place patch (no recreate)
./deploy-hlh-ai-engine-vllm.sh --destroy      # full rebuild
```

The deploy script:
1. **Validates the host NVIDIA R580 580.65.06 driver + V100** (the host driver install itself is owned by the sibling `hlh-ai-engine-egpu` repo — if the driver is missing it points you there).
2. Creates privileged LXC **113** (`hlh-ai-engine-vllm`) at **192.168.1.13** on the `RaidZ1-6TB` pool (48 GB RAM, 12 cores, 64 GB rootfs).
3. Wires V100 CUDA passthrough: `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`, `/dev/nvidia-uvm-tools`, `/dev/nvidia-modeset` + cgroup2 allows (`195:*`, `507:*`, `508:*`, `510:*`, `511:*`).
4. Pushes `configure-hlh-ai-engine-vllm.sh` and runs it inside the LXC: NVIDIA driver userspace (CUDA 12.8 branch) + **vLLM 0.19.1 (PyPI CUDA build)** + **Open WebUI**, all via `uv` venvs.

## What this is

```
prox01 (Proxmox VE, kernel 6.14.11-9-pve)
└── LXC 113 hlh-ai-engine-vllm (192.168.1.13, privileged, ubuntu-24.04)
    ├── vLLM 0.19.1 (native, /opt/vllm-venv, torch 2.10.0+cu128) — OpenAI API :8000
    ├── Open WebUI (native, /opt/open-webui-venv) — chat UI :80
    ├── /dev/nvidia0 + UVM (V100 GV100 32GB, Volta sm_70, via OCuLink c5:00.0)
    │                 └─ nvidia-smi reports this board as "Tesla PG500-216" (VBIOS name), not "Tesla V100"
    └── /srv/ai/models → host /srv/ai/models (ZFS RaidZ1-6TB, shared with LXC 111)
```

The **V100 GV100GL 32GB HBM2** is passed through to this LXC **and** to LXC 111 (`hlh-ai-engine-egpu`, llama.cpp) — the NVIDIA driver multiplexes, but the 32 GB of VRAM is shared. See [GPU co-tenancy](#gpu-co-tenancy-with-hlh-ai-engine-egpu-lxc-111).

## Pinned stack — why vLLM 0.19.1 (Volta sm_70 ceiling)

The V100 is **Volta, compute capability 7.0**. The stack is capped by the driver line:

| Layer | Pin | Why |
|---|---|---|
| Host driver | **NVIDIA 580.65.06 (R580)** | Last driver branch that supports Volta. R590+ dropped Volta entirely. Owned by `hlh-ai-engine-egpu`. |
| Kernel | proxmox-kernel-**6.14.11-9-pve** (LTS) | 6.17/7.0 pve kernels break the 550/580 closed DKMS (see egpu README). |
| CUDA | **12.8** | CUDA 13.0 dropped Volta (sm_70). 12.8 is the last CUDA with sm_70 kernels. |
| torch | **2.10.0** (PyPI default = cu12.8) | Pulls the CUDA 12.8 runtime via `nvidia-*-cu12` pip packages — no CUDA toolkit apt packages needed. |
| vLLM | **0.19.1** (PyPI) | **Last stable with the CUDA 12.8 / torch 2.10.0 build.** vLLM 0.20.0+ pins torch 2.11.0+cu13 → no sm_70 → cannot run on the V100. |

So: do **not** upgrade vLLM past 0.19.1 on this GPU, and do **not** move the host driver to R590+. `wheels.vllm.ai` does not host stable CUDA wheels — the CUDA build is the plain PyPI `vllm` wheel (the `rocm`/`xpu` builds come from `wheels.vllm.ai`, which this repo no longer uses).

## Files

| File | Purpose |
|---|---|
| `deploy-hlh-ai-engine-vllm.sh` | Host-side: validate R580/V100, create LXC 113, GPU passthrough, run bootstrap, verify. |
| `configure-hlh-ai-engine-vllm.sh` | Inside LXC: driver userspace (580 branch, exact match to host kernel driver, CUDA 12.8), uv venvs, vLLM 0.19.1 (PyPI CUDA) + Open WebUI, systemd units, hard verification. |
| `00_BACKLOG.md` | Ideas / later work. |
| `10_ACTIVE.md` | Current focus. |
| `90_DONE.md` | Completed work. |
| `checkpoint.md` | Resume state + historical notes (ROCm 890M era, CUDA 13 / vLLM 0.20 investigation). |

## Usage

```bash
# Health + served models
curl -s http://192.168.1.13:8000/health && curl -s http://192.168.1.13:8000/v1/models | head -100

# Chat UI (no Docker; native on port 80)
open http://192.168.1.13/

# GPU state (shared 32GB with LXC 111)
ssh root@192.168.1.13 'nvidia-smi'

# Switch the served model (interactive: dir name, path, or HF id)
ssh root@192.168.1.13 /usr/local/bin/vllm-switch-model.sh

# Logs
ssh root@192.168.1.13 'journalctl -u vllm -f'
ssh root@192.168.1.13 'journalctl -u open-webui -f'

# Runtime config
ssh root@192.168.1.13 'cat /etc/vllm.env'
ssh root@192.168.1.13 'cat /etc/open-webui.env'
```

Default model: `/srv/ai/models/Qwen3.6-35B-A3B-GPTQ-Int4` (served as `qwen3.6-35b-a3b-gptq-int4`). Tune `/etc/vllm.env`: `AI_GPU_MEM_UTIL` (default 0.85), `AI_MAX_MODEL_LEN` (default 16384), `AI_API_KEY` (set it — the API is bound 0.0.0.0), `AI_EXTRA_ARGS`.

## GPU co-tenancy with hlh-ai-engine-egpu (LXC 111)

The same OCuLink V100 (`c5:00.0`) is passed through to **both** LXCs. Both engines can run simultaneously, but they share the **32 GB HBM2**:

- The egpu engine (llama.cpp) at its default config (27B Q4 + 128K KV) uses ~33 GB — effectively the whole card.
- If you want vLLM on the V100, **stop LXC 111's server** (`pct exec 111 -- systemctl stop llama-server`) or right-size it (shorter context, smaller model) and lower vLLM's `AI_GPU_MEM_UTIL` (e.g. `0.45`).
- `nvidia-smi` (host or LXC) shows the combined VRAM usage of both.

## V100 performance notes (Volta sm_70)

- **fp16 compute** — vLLM avoids bf16 on CC < 8.0 (`dtype=auto` resolves to fp16 on the V100). bf16-only checkpoints get cast; prefer fp16/GPTQ/Int8 checkpoints.
- **`--enforce-eager`** is on by default in `/usr/local/bin/vllm-run.sh` (CUDA-graph capture on Volta can be flaky). Remove it there if you want to try graphing.
- **Attention backend**: vLLM 0.19.1 picks **`TRITON_ATTN`** on sm_70 (FA2/FlashInfer need cc ≥ 8.0; only the multimodal encoder uses SDPA). Consequence: the LXC needs a **C compiler (`gcc`)** — Triton JIT-compiles its C driver extension on first kernel launch. Configure installs `gcc`/`g++`, gates on `cc`, and pre-warms the Triton JIT cache at deploy. If you ever see `RuntimeError: Failed to find C compiler`, that's what's missing.
- **GPTQ-Int4**: Marlin GPTQ kernels require SM80+; on the V100 vLLM falls back to the standard GPTQ kernels (slower, works).
- **No FP8, no FlashInfer** on sm_70 — expected; avoid FP8 checkpoints.
- 32 GB HBM2 comfortably holds the 35B-A3B GPTQ-Int4 (~20 GB) with a 16K context at 0.85 util; MoE A3B (≈3B active) gives good tok/s on the V100's 900 GB/s.

## Troubleshooting

- **`nvidia-smi` shows `Tesla PG500-216` instead of `Tesla V100`** → that **is** the V100. The GV100GL board reports its VBIOS product name, not the marketing name (documented in `hlh-ai-engine-egpu` too). The gate is compute capability 7.0, not the string — don't "fix" it.
- **`nvidia-smi: command not found` in LXC** → driver userspace missing; re-run configure (installs `libnvidia-compute-580` + `nvidia-utils-580`).
- **`Failed to initialize NVML: Driver/library version mismatch`** → the LXC userspace version does not exactly match the host kernel driver (NVIDIA rotates 580-branch point releases). Re-run configure — it resolves the host driver's exact version via `apt-cache madison`, unholds, and re-pins the full 5-package set (`libnvidia-compute/cfg1/decode/gpucomp-580`, `nvidia-utils-580`). If the repo no longer carries that driver version, upgrade the host driver to the current 580 tip (`hlh-ai-engine-egpu`) and re-run both.
- **FATAL "not enough free VRAM on the shared V100"** → LXC 111 (llama.cpp) is holding VRAM on the same card: `pct exec 111 -- systemctl stop ai-engine`, then re-run (or `SKIP_VRAM_PREFLIGHT=1` / lower `AI_GPU_MEM_UTIL` for co-tenancy).
- **`RuntimeError: Failed to find C compiler` (EngineCore dies in `profile_run`, service crash-loops)** → no `cc` in the LXC: on sm_70 vLLM uses the TRITON_ATTN backend + Triton kernels (ViT rotary, GDN prefill), and Triton JIT-compiles a C driver extension on first launch. Fix in the LXC: `apt-get update && apt-get install -y gcc g++ && systemctl restart vllm` (configure ≥ 0.6.2 installs + gates on gcc and pre-warms the Triton cache; first serve after the fix compiles kernels for a couple of minutes).
- **`torch.cuda.is_available() == False`** → `/dev/nvidia0`/`/dev/nvidiactl`/`/dev/nvidia-uvm` not bound; redeploy (passthrough block) and restart LXC.
- **`no kernel image is available for execution on the device`** → you're running a cu13/sm_75+ build; reinstall `vllm==0.19.1` (cu128).
- **VRAM OOM while LXC 111 is running** → shared card; stop/resize the sibling engine or lower `AI_GPU_MEM_UTIL`.
- **Host driver missing / wrong branch** → `cd ~/git/hlh-ai-engine-egpu && ./deploy-hlh-ai-engine-egpu.sh` (owns the R580 580.65.06 install on 6.14 LTS kernel).

## Repo layout

```
hlh-ai-engine-vllm/
├── deploy-hlh-ai-engine-vllm.sh
├── configure-hlh-ai-engine-vllm.sh
├── README.md
├── CHANGELOG.md
├── checkpoint.md
├── 00_BACKLOG.md
├── 10_ACTIVE.md
└── 90_DONE.md
```
