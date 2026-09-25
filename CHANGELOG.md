# Changelog

All notable changes to this repository are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.2] - 2026-09-25

### Fixed
- **vLLM crash-looped with `RuntimeError: Failed to find C compiler`** (3rd live run): on sm_70 vLLM 0.19.1 selects the **TRITON_ATTN** attention backend (FA2 needs cc ≥ 8.0) plus Triton kernels for the ViT rotary + GDN prefill; Triton JIT-compiles a small C driver extension on the **first** kernel launch, which needs `cc`. The LXC base install (`--no-install-recommends: curl ca-certificates git openssh-server gnupg python3 libgomp1`) had no compiler, so EngineCore died during the KV-cache `profile_run` (dummy multimodal pass) and systemd restart-looped — the model weights themselves loaded fine (21 GiB / 23 s).
- Configure now installs **`gcc g++ python3-dev`** with the base packages and **hard-gates on `cc` + `Python.h`** (FATAL with the one-line fix instead of a cryptic runtime crash). `python3-dev` is required too: with `cc` present but no headers, the same first Triton launch dies with `fatal error: Python.h: No such file or directory` (2nd live failure, caught by the new smoke test).
- [4/8] verification now runs a **real Triton JIT compile + launch** (trivial `tl.store` kernel on sm_70) — catches the missing-compiler class of failure at deploy time and **pre-warms the `/root/.triton` cache** so the first `vllm serve` doesn't pay the compile tax.

### Docs
- README: corrected the attention-backend note (sm_70 → **TRITON_ATTN**, not "falls back to SDPA" — only the MM encoder uses SDPA) and added a `Failed to find C compiler` troubleshooting entry with the in-LXC fix.
- Immediate fix for an already-deployed LXC: `apt-get update && apt-get install -y gcc g++ python3-dev && systemctl restart vllm`.

## [0.6.1] - 2026-09-24

### Fixed
- **580-branch apt rotation broke the pinned userspace install** (live 2nd run): NVIDIA rotated the 580 point release (580.65.06 → 580.178.04) in the CUDA ubuntu2404 repo; a transitional apt index made the `580.65.06-0ubuntu1` pin unresolvable, and the script's `||` unpinned fallback then silently installed the **mismatched** 580.178.04 userspace → `Failed to initialize NVML: Driver/library version mismatch` (NVML requires userspace == host kernel driver version exactly).
- Configure now **resolves the exact package version string** for the host driver via `apt-cache madison` (handles revision suffixes: `580.65.06-0ubuntu1` vs `580.178.04-1ubuntu1`) and pins the **full 5-package set** (`libnvidia-compute-580`, `libnvidia-cfg1-580`, `libnvidia-decode-580`, `libnvidia-gpucomp-580`, `nvidia-utils-580`) with `--allow-downgrades` — which also heals an already-mixed LXC. **No unpinned fallback**: pin failure is a loud FATAL with the available versions + guidance (repo rotated → upgrade host driver via `hlh-ai-engine-egpu`; repo mid-rotation → `apt-get update` + re-run).
- `apt-mark unhold` before re-pinning (idempotent re-runs); removes `nvidia-persistenced` if present (branch-locked, not needed in the LXC).
- Post-install NVML gate now diagnoses version mismatches explicitly (prints host kernel vs LXC userspace versions + fix command).

### Notes
- Verified against the live CUDA repo index: `580.65.06-0ubuntu1` is listed and downloadable for all five packages after the rotation settled — the failure was a mid-rotation race compounded by the fallback.
- `hlh-ai-engine-egpu` (LXC 111) got the same hardening: `unhold` before re-pin + post-install FATAL if `libnvidia-compute-580` != host driver version. LXC 111 is unaffected (its userspace already matches).

## [0.6.0] - 2026-09-24

### Changed
- **GPU backend refactor: AMD 890M (ROCm 7.2) → NVIDIA Tesla V100 GV100 32GB eGPU (CUDA 12.8, OCuLink `c5:00.0`)**, mirroring the `hlh-ai-engine-egpu` stack:
  - Host: NVIDIA **R580 580.65.06** (last driver branch for Volta) on `proxmox-kernel-6.14.11-9-pve` — deploy now *validates* this (host driver install stays owned by `hlh-ai-engine-egpu`); `--skip-host-driver` for re-runs.
  - LXC 113 passthrough: `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`, `/dev/nvidia-uvm-tools`, `/dev/nvidia-modeset` + cgroup2 allows `195:* 507:* 508:* 510:* 511:*`.
  - LXC userspace: `libnvidia-compute-580` + `nvidia-utils-580` (580.65.06-0ubuntu1, CUDA ubuntu2404 repo) — no CUDA toolkit (wheels bundle the CUDA 12.8 runtime).
  - Removed all ROCm machinery: `wheels.vllm.ai` index, `rocm` apt repo/pin/keyring, `openmpi`, `HIP_VISIBLE_DEVICES`, `FLASH_ATTENTION_TRITON_AMD_ENABLE`, `aiter`/`triton-mlir`/`pyrsmi`.
- **vLLM pin 0.30.0 → 0.19.1** (PyPI CUDA build): 0.19.1 is the **last stable with torch 2.10.0+cu128 (sm_70 kernels)**; vLLM 0.20.0+ pins torch 2.11.0+cu13 and CUDA 13.0 dropped Volta — it cannot run on the V100. Verification now asserts torch cu12.8, a single Tesla V100, a real fp16 kernel launch on sm_70, and fails on any `+rocm`/`nvidia-*-cu13` package in the venv.
- Runtime defaults: `AI_GPU_MEM_UTIL=0.85` (dedicated 32 GB HBM2), `AI_MAX_MODEL_LEN=16384`; `--enforce-eager` kept (Volta-safe).

### Fixed
- Deploy/configure no longer FATAL on the board's VBIOS name: this GV100GL reports **"Tesla PG500-216"** in `nvidia-smi -L`, not "Tesla V100". Gates are now exactly-one-GPU (host) + torch compute-capability (7,0) (in LXC); the name is informational.
- New VRAM preflight in configure: FATAL (with remediation `pct exec 111 -- systemctl stop ai-engine`) when the shared 32 GB has less than `AI_GPU_MEM_UTIL` free; `SKIP_VRAM_PREFLIGHT=1` to override.

### Added
- README: Volta sm_70 stack-ceiling table (driver R580 / CUDA 12.8 / torch 2.10 / vLLM 0.19.1), GPU co-tenancy section (V100 shared with LXC 111 llama.cpp — 32 GB VRAM is shared), V100 performance notes (fp16, GPTQ non-Marlin fallback, no FP8/FlashInfer on sm_70).

## [0.5.1] - 2026-09-20

### Fixed

- **CRITICAL: torch import failed with `ImportError: libroctx64.so.4` — ROCm libs never installed in LXC** — root cause: the ubuntu-24.04 LXC template ships **without `gnupg`**, so `wget rocm.gpg.key | gpg --dearmor | tee amdrocm.gpg` silently wrote a **0-byte keyring** (every stage masked by `> /dev/null 2>&1` / `|| true`). `apt-get update` then failed `GPG error: NO_PUBKEY 9386B48A1A693C5C` (swallowed by `|| true`), all three `apt-get install rocm…` fallbacks died with `Unable to locate package`, the script warned and continued, and the [4/8] torch gate died on missing `libMIOpen/libamdhip64/libroctx64/…`. Now: `gnupg` added to base packages with a hard `command -v gpg` gate; new `fetch_rocm_key()` retries the fetch, dearmors with `gpg --batch --yes`, and **fails loudly** unless the keyring is non-empty AND parseable (`gpg --show-keys`).
- **ROCm apt repo selection now probed, not guessed** — 0.5.0 wrote the repo line blind (and its `7.2.3` fallback was a dead `echo A > file || echo B > file` that could never fire). Now probes `https://repo.radeon.com/rocm/apt/<dotted|major.minor>/dists/<codename>/Release` (dotted `7.2.3` first for exact wheel match) and the 10.x `stable.repo.amd.com` dists before writing `rocm.list`; FATAL with probed-URL list if none reachable.
- **`apt-get update` for ROCm repo no longer swallowed** — logged to `/tmp/apt-update-rocm.log`, and a **candidate check** (`apt-cache policy rocm-core`) is a hard gate before any install attempt.
- **Individual-libs fallback list corrected** — dropped nonexistent `hiprtc-amd` (libhiprtc ships in `hip-runtime-amd`), added `hsa-amd-aqlprofile` (`libhsa-amd-aqlprofile64.so.1`) and `libdw1` (`libdw.so.1`) — both are torch-wheel deps that only surface after the first 15 libs are present. Individual install failure is now FATAL (was warn-and-continue into a guaranteed torch crash ~7 min later).
- **Early lib-resolution gate after ROCm install** — `libamdhip64/libroctx64/librocblas/libMIOpen` must resolve via `ldconfig -p` immediately after apt install (fail in ~1 min, not after the full pip install); the [4/8] `ldd torch/lib/*.so` missing-libs check is now FATAL instead of WARNING.
- **No more hardcoded `/opt/rocm-7.2.0`** — `ROCM_LIB_DIRS` / `_ROCM_HOME` derived from the `/opt/rocm*` dirs that actually exist (7.2.3 installs to `/opt/rocm-7.2.3`); `/etc/ld.so.conf.d/rocm.conf` + `LD_LIBRARY_PATH` in the verify step and `vllm-run.sh` use the derived paths.
- **Deploy (host path)** — same silent-failure class: host ROCm key fetch now verifies the keyring is non-empty before trusting the repo (FATAL otherwise).

## [0.5.0] - 2026-09-20

### Changed

- **Simplicity-first native stack: `uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/0.29.0/rocm723`** — configure now auto-resolves **latest stable vLLM + matching ROCm variant** at deploy time from `wheels.vllm.ai` (today `0.29.0/rocm723` → ROCm 7.2.3, Python 3.12/cp312). Zero script changes when `0.30.0` ships. Same resolver in `deploy` and `configure` so CT build and LXC bootstrap always agree.
- **Open WebUI included natively (no docker)** — same LXC, two venvs (`/opt/vllm-venv` + `/opt/open-webui-venv`), single `open-webui.service` on **port 80** (no port mapping). `tok/s` first: native HIP avoids docker veth/IPC overhead on the 890M APU's UMA/GTT path. WebUI talks to vLLM at `127.0.0.1:8000/v1` with `BYPASS_MODEL_ACCESS_CONTROL=true`, data at `/opt/open-webui/data`.
- **Host ROCm stays in sync** — deploy translates variant `rocm723` → `7.2.3` and prompts to upgrade host if major mismatches or `/dev/kfd` missing. Wheels bundle userspace; host only needs amdgpu kernel driver/firmware. `ROCM_VERSION` legacy env still honored (derived to `VLLM_ROCM_VARIANT`), but `VLLM_VERSION`/`VLLM_ROCM_VARIANT` are now canonical. `HOST_ROCM_SETUP=0` skips prompt.
- **Removed:** all Docker, ROCm apt in LXC, `ROCM_VERSION=10.0.0` hard-pin, `AMD_SMI_PATH` PYTHONPATH hack, `TORCH_SPEC`/`VLLM_WHEEL` hard-pins, `stable.repo.amd.com/whl-next` + `rocm.frameworks.amd.com` URLs, `VLLM_USE_V2_MODEL_RUNNER` / `HSA_OVERRIDE_GFX_VERSION` mandatory flags (now commented last-resort).

### Added

- `open-webui.service` (native, port 80) + `/etc/open-webui.env` + `/usr/local/bin/open-webui-run.sh` — enable with `systemctl enable --now open-webui`, logs `journalctl -u open-webui -f`.
- Resolver helpers `resolve_vllm_version()` / `resolve_rocm_variant()` / `variant_to_dotted()` / `resolve_python_version()` in both scripts — single source of truth via PyPI JSON + `wheels.vllm.ai` HTML probe.
- `vllm` install now via `uv pip install vllm --extra-index-url $WHEELS` (pip local-version wins `+rocm723` over PyPI CUDA), optional `flash-attn`/`amd-aiter` warn-and-continue with `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` fallback.

## [0.3.3] - 2026-09-18

### Fixed

- **CRITICAL: vllm_c rms_norm missing _C** — `AttributeError: _C has no attribute rms_norm` at `vllm/kernels/vllm_c.py:43` → `layernorm.py:115` during `profile_run` (CUDA wheel has no `vllm._C`/`_rocm_C`, `ldd` shows `libcudart.so.13` missing). Previous regex gating `GPGPU_DEVICE` failed due to parentheses mismatch. Now: adds `_c_ext_available()` helper, gates `GPGPU_DEVICE` correctly, adds native fallback `ir.ops.rms_norm.impls["native"]` and `layernorm.py` `forward_cuda` early return to `forward_native` when `_C.rms_norm` missing (P1 immediate unblock). Keeps native alive for your test.
- **Triton 3.4.0 vs 3.7 `target_info`/`constexpr_function` mismatch** — `pytorch-triton-rocm 3.4.0` lacks `triton.language.target_info` needed by `vllm 0.29.0` `third_party/triton_kernels/target_info.py:1` (`Failed to import Triton kernels`), while `3.7.1` broke `constexpr_function`. Now: shim `triton/language/target_info.py` with minimal `is_cuda/is_hip/cuda_capability_geq` etc., and alias `constexpr_function` in `triton/runtime/jit.py` when missing (P2). Reduces fragility per review.

## [0.3.2] - 2026-09-18

### Fixed

- **CRITICAL: amdsmi pip vs lib ABI mismatch** — `pip amdsmi 7.0.2` vs system `libamd_smi.so 27.0.0` (ROCm 10.0) `AttributeError: undefined symbol: amdsmi_set_gpu_clk_range` at `amdsmi/amdsmi_wrapper.py:2642` triggered via `torch/cuda/__init__.py:110 import amdsmi` and `vllm_rocm_accel_shim.pth:1` (Grok correctly diagnosed pip vs system version skew). Now: uninstall pip `amdsmi`, reinstall from `/opt/rocm/share/amd_smi` (guaranteed ABI match) when present, else leave uninstalled and rely on `vllm/platforms/__init__.py:411` HIP fallback; verify `import torch` succeeds before proceeding. Live LXC `113` was on restart loop 94 due to this.

## [0.3.1] - 2026-09-18

### Fixed

- **CRITICAL: vLLM V2 runner UVA crash** — `AttributeError: torch.ops._C.get_cuda_view_from_cpu_tensor` at `vllm/utils/torch_utils.py:916` → `vllm/v1/worker/gpu/buffer_utils.py:50` (GPUModelRunnerV2). PyPI `vllm 0.29.0` CUDA wheel has no ROCm UVA op on `torch 2.8.0+rocm6.4`. Fix: set `VLLM_USE_V2_MODEL_RUNNER=0` in `/etc/vllm.env`, `vllm-run.sh`, and `vllm.service` (`Environment=VLLM_USE_V2_MODEL_RUNNER=0`) — proven `checkpoint.md:4-5`.
- **ROCm paths** — runner hardcoded `/opt/rocm/core-10.0` but live LXC uses `/opt/rocm` symlink (`/etc/alternatives/core`). Now uses `/opt/rocm/lib` + `PATH=/opt/rocm/bin`; `ld.so.conf` covers both.
- **Triton pin** — `pytorch-triton-rocm 3.4.0` vs `triton 3.7.1` mismatch → `cannot import constexpr_function`. Now force `triton==3.4.0` after venv.
- **Deploy UX** — `deploy-hlh-ai-engine-vllm.sh` now offers `y` (destroy/recreate) / `u` (update in-place, fast ~2-5 min) / `n` (abort) when LXC exists; flags `--update` / `--destroy` for non-interactive.

### Changed

- `VLLM_MAX_MODEL_LEN` reverted to `4096` (was 131072 in 0.3.0; 131k fails validation for Qwen3.5-9B, derived max ~40960; 4096 matches `checkpoint.md` proven safe and commit `c5f4a39`).

## [0.3.0] - 2026-09-18

### Changed

- **Phase 2 refactor: REMOVED DOCKER. vLLM now runs natively in LXC** (no docker container layer). Bootstrap installs ROCm userspace in-container (matching host ROCM_VERSION), Python venv at `/opt/vllm-venv`, vLLM via pip with ROCm support. `vllm.service` `ExecStart` is now `/usr/local/bin/vllm-run.sh` (native python -m vllm.entrypoints.openai.api_server).
- `deploy-hlh-ai-engine-vllm.sh` no longer references docker; forwards `ROCM_VERSION` and `VLLM_DEFAULT_MODEL` to bootstrap; backend description updated to "vLLM ROCm native".
- Runtime config `/etc/vllm.env` simplified (removed `VLLM_IMAGE`, `VLLM_SHM_SIZE`, docker-specific vars); added native ROCm env vars.
- Default model confirmed as **`/srv/ai/models/Qwen3.5-9B`** (18 GB bf16 safetensors, `qwen3_5` VLM, served as `qwen3.5-9b`).
- `HSA_OVERRIDE_GFX_VERSION=11.0.0` remains mandatory (proven override for gfx1150).

### Removed

- Docker installation and `docker pull`/`docker run` logic from bootstrap.
- `vllm-docker-run.sh` runner; replaced by `vllm-run.sh` (native).
- `--security-opt apparmor=unconfined`, `--security-opt seccomp=unconfined`, `--group-add`, `--ipc host`, `--shm-size` docker flags.
- `VLLM_IMAGE` env var and docker image references throughout.
- Open WebUI remains deferred to **phase 10**.

### Added

- In-container ROCm userspace installation (matching host major version via `stable.repo.amd.com` for 10.x or `packages-multi-arch` for 7.x).
- Python venv creation at `/opt/vllm-venv` with torch (ROCm index) and vLLM[rocm] installation.
- ROCm compatibility patches applied at bootstrap: torch.accelerator shim, SiluAndMul native fallback, vllm_c provider gating (from `checkpoint.md` proven fixes).
- `README.md` fully rewritten for native architecture (runtime contract, tuning table, health checks, gotchas).
- `deploy-hlh-ai-engine-vllm.sh` usage text updated for native path.

## [0.2.0] - 2026-09-16

### Changed

- **Phase 1 refactor: vLLM now runs as the official docker image `vllm/vllm-openai-rocm:latest`** (ROCm userspace ships in the image). `vllm.service` `ExecStart` is now `/usr/local/bin/vllm-docker-run.sh` (docker run, `--network host`, `--ipc host`, `--shm-size 16g`, `--device /dev/kfd /dev/dri/renderD128 /dev/dri/card0`, numeric `--group-add`, `--security-opt apparmor=unconfined seccomp=unconfined`, model dir mounted read-only at `/srv/ai/models`; the runner pulls `VLLM_IMAGE` at start so `/etc/vllm.env` is the single source of truth)
- Runtime config moved to **`/etc/vllm.env`** (`VLLM_IMAGE`, `VLLM_PORT`, `VLLM_MODEL_PATH`, `VLLM_SERVED_NAME`, `VLLM_GPU_MEM_UTIL`, `VLLM_MAX_MODEL_LEN`, `HSA_OVERRIDE_GFX_VERSION`, `VLLM_SHM_SIZE`, `VLLM_LOG_LEVEL`, `VLLM_EXTRA_ARGS`) — edit + `systemctl restart vllm`
- Default model now **`/srv/ai/models/Qwen3.5-9B`** (18 GB bf16 safetensors, `qwen3_5` hybrid linear-attention VLM, served as `qwen3.5-9b`); `--gpu-memory-utilization 0.40`, `--max-model-len 4096`, `--enforce-eager`, `--trust-remote-code`, `--limit-mm-per-prompt '{"image":1,"video":1}'`
- `HSA_OVERRIDE_GFX_VERSION` default back to **`11.0.0`** (the proven override on this box; `11.5.0` → `HIP error: invalid device function`, see `checkpoint.md` §4.5) — now set as container env, no in-LXC ROCm/profile.d needed
- `vllm-switch-model.sh` now edits `/etc/vllm.env` (local dirs or HF ids) instead of sed-ing ExecStart
- `deploy-hlh-ai-engine-vllm.sh` forwards `VLLM_IMAGE` (not `ROCM_VERSION`) into the bootstrap; host ROCm logic kept (amdgpu kernel driver/firmware for `/dev/kfd`); usage text fixed (said LXC 112, is 113); deploy notes that sibling `112` is never touched

### Removed

- In-LXC ROCm apt install, `/opt/vllm-venv`, `pip vllm==0.29.0` + ROCm torch re-install dance, and all wheel shims/patches (`vllm_rocm_accel_shim`, `torch_c_dlpack_ext` patch, `vllm_c` gating, `libtorch_cuda` symlink) — superseded by the official ROCm image
- Open WebUI from the phase-1 bootstrap (`open-webui.service` no longer created) — deferred to **phase 10**
- Stale `Qwen3.5-9B-safetensors` / `qwen3.5-9b 0.70` references throughout

### Added

- `00_BACKLOG.md`: explicit phase-10 Open WebUI section + docker data-root subvolume idea
- `README.md`: runtime contract, `/etc/vllm.env` tuning table, container flag notes, co-tenancy policy (never stop `112`)

## [0.1.0] - 2026-09-11

### Added

- Initial `hlh-ai-engine-vllm` forked from `hlh-ai-engine` `f73bece` (`v0.9.4` `10.0.0` `HIP+Vulkan`)
- LXC `113` `hlh-ai-engine-vllm` `192.168.1.13/24` on `RaidZ1-6TB` `48G/12c/64G` `prox01`, privileged `nesting,keyctl`, `onboot 1`
- `deploy-hlh-ai-engine-vllm.sh` — host ROCm upgrade prompt (`7.14.0 → 10.0.0` via `stable.repo.amd.com/debian13` or `packages-multi-arch`, `trixie→debian13` fix, `Acquire::Check-Valid-Until=false`), GPU passthrough `226:1/226:129/511:0` (fixed `none dev/dri` host-root mount bug), `113`/`192.168.1.13`, `VLLM ROCm + Open WebUI` backend
- `configure-ai-engine-inside-lxc.sh` `0.1.0` — replaces `llama.cpp` `cmake GGML_HIP+VULKAN` with Python venv `/opt/vllm-venv` `pip vllm[rocm]` (`HSA_OVERRIDE_GFX_VERSION=11.5.0`) + `docker.io` `open-webui` (`ghcr.io/open-webui/open-webui:main` `--network host` `OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1` `:8080`)
- Services: `vllm.service` `:8000` (`Qwen/Qwen2.5-Coder-32B-Instruct` `qwen2.5-coder-32b` `gpu_memory_utilization 0.85` `enforce_eager`) + `open-webui.service` `:8080` (`BYPASS_MODEL_ACCESS_CONTROL=true`), `Restart=on-failure`
- `vllm-switch-model.sh` (`/usr/local/bin` + `/srv/ai/models/vllm-switch-model.sh`) for HF model ID switching
- `configure-hlh-ai-engine-vllm.sh` (pure bash `pct push`/`pct exec`, `192.168.1.13`) — reconfigure path
- LXC `113` `192.168.1.13/24` pure bash `deploy-hlh-ai-engine-vllm.sh` + `README` `vLLM :8000` + `Open WebUI :8080` runtime contract
- Shared `/srv/ai/models` bind mount (`mp0`) with siblings (`112` GGUF, `131` CUDA, `113` HF at `/.hf-cache`)

### Known limitations

- `gfx1150` `890M` APU is community-tier for `vLLM` ROCm — may need `--enforce-eager` (APU lacks some flash attn kernels); `hlh-ai-engine` `llama.cpp` remains the stable 890M path until vLLM 890M validated
- `112` (`hlh-ai-engine`) and `113` (`hlh-ai-engine-vllm`) share same `c9:00.0` `150e` — stop one before starting the other (`pct stop 112 && pct start 113`)
