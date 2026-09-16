# Changelog

All notable changes to this repository are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- `ansible/files/configure-ai-engine-inside-lxc.sh` `0.1.0` — replaces `llama.cpp` `cmake GGML_HIP+VULKAN` with Python venv `/opt/vllm-venv` `pip vllm[rocm]` (`HSA_OVERRIDE_GFX_VERSION=11.5.0`) + `docker.io` `open-webui` (`ghcr.io/open-webui/open-webui:main` `--network host` `OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1` `:8080`)
- Services: `vllm.service` `:8000` (`Qwen/Qwen2.5-Coder-32B-Instruct` `qwen2.5-coder-32b` `gpu_memory_utilization 0.85` `enforce_eager`) + `open-webui.service` `:8080` (`BYPASS_MODEL_ACCESS_CONTROL=true`), `Restart=on-failure`
- `vllm-switch-model.sh` (`/usr/local/bin` + `/srv/ai/models/vllm-switch-model.sh`) for HF model ID switching
- `ansible/inventories/hlh-ai-engine-vllm.yml` `192.168.1.13` `hlh_ai_engine_vllm` + `ansible/playbooks/hlh-ai-engine-vllm.yml` + `configure-hlh-ai-engine-vllm.sh`
- `opentofu/` `vmid 113` `192.168.1.13/24` `hlh-ai-engine-vllm` + `README` `vLLM :8000` + `Open WebUI :8080` runtime contract
- Shared `/srv/ai/models` bind mount (`mp0`) with siblings (`112` GGUF, `131` CUDA, `113` HF at `/.hf-cache`)

### Known limitations

- `gfx1150` `890M` APU is community-tier for `vLLM` ROCm — may need `--enforce-eager` (APU lacks some flash attn kernels); `hlh-ai-engine` `llama.cpp` remains the stable 890M path until vLLM 890M validated
- `112` (`hlh-ai-engine`) and `113` (`hlh-ai-engine-vllm`) share same `c9:00.0` `150e` — stop one before starting the other (`pct stop 112 && pct start 113`)
