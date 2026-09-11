# Changelog

All notable changes to this repository are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
