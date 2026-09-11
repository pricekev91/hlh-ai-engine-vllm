# DONE

This is what is already implemented and verified in this repository.

## LXC Deployment

- Direct Proxmox LXC creation via `deploy-hlh-ai-engine-vllm.sh` — **LXC 113** (`.13` parity with `192.168.1.13`, `Proxmox >=100`)
- Privileged LXC `113` `hlh-ai-engine-vllm` `48 GiB RAM (49152)` `12 cores` `64G rootfs` on `RaidZ1-6TB` `onboot 1` `vmbr0` `192.168.1.13/24` gw `192.168.1.1`
- Host ROCm upgrade prompt before LXC (checks `get_host_rocm_version` vs `ROCM_VERSION` `10.0.0` default never pinned, switches `stable.repo.amd.com` for `10.x`)
- Prompt-before-redeploy guard, GPU passthrough `226:1/226:129/511:0` (`gfx1150` only)

## GPU Passthrough

- Same as `hlh-ai-engine` `112`: `/dev/dri/card1` `renderD129` + `/dev/kfd` (`gfx1150` Strix Halo, `c9:00.0` `150e` APU)

## ROCm / Runtime

- ROCm `10.0.0` default never pinned (host must match LXC major), `ROCM_VERSION=7.14.1` still via env
- Python venv `/opt/vllm-venv` with `vllm[rocm]` (pip, `HSA_OVERRIDE_GFX_VERSION=11.5.0`)
- Docker `docker.io` for Open WebUI

## Services

- `vllm.service` `systemd` `WorkingDirectory=/srv/ai/models` `ExecStart=/opt/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-Coder-32B-Instruct --host 0.0.0.0 --port 8000 --served-model-name qwen2.5-coder-32b --gpu-memory-utilization 0.85 --max-model-len 8192 --enforce-eager --dtype half` `Restart=on-failure`
- `open-webui.service` `systemd` `docker run --network host -v open-webui:/app/backend/data -e PORT=8080 -e OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1 ghcr.io/open-webui/open-webui:main` `Restart=on-failure`
- `vllm-switch-model.sh` (`/usr/local/bin` + `/srv/ai/models/vllm-switch-model.sh`) rewrites `--model`/`--served-model-name` and health-probes `/health`

## Model Management

- HF default `Qwen/Qwen2.5-Coder-32B-Instruct` (`qwen2.5-coder-32b`), `dl-hf.sh` helper for HF Hub, cache at `/srv/ai/models/.hf-cache`
- Shared `/srv/ai/models` bind mount with siblings (`112` GGUF, `113` HF)

## Networking

- vLLM API `http://192.168.1.13:8000` (`/health`, `/v1/models`, `/v1/chat/completions`) — OpenAI-compatible
- Open WebUI `http://192.168.1.13:8080` (chat UI)

## Ansible

- `ansible/inventories/hlh-ai-engine-vllm.yml` `192.168.1.13` `hlh_ai_engine_vllm` `root`
- `ansible/playbooks/hlh-ai-engine-vllm.yml` `ansible.builtin.script` → `configure-ai-engine-inside-lxc.sh`

## OpenTofu

- `telmate/proxmox >= 2.7.2` `proxmox_lxc hlh_ai_engine_vllm` `vmid 113` `192.168.1.13/24`

## Verification

- `rocm-smi` `hipconfig` `vllm --help` `docker ps` `systemctl status vllm/open-webui` `curl :8000/health`
