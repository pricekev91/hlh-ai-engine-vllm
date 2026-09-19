# Changelog

All notable changes to this repository are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
