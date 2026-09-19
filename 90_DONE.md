# DONE

This is what is already implemented and verified in this repository.

## LXC Deployment

- Direct Proxmox LXC creation via `deploy-hlh-ai-engine-vllm.sh` — **LXC 113** (`.13` parity with `192.168.1.13`, `Proxmox >=100`)
- Privileged LXC `113` `hlh-ai-engine-vllm` `48 GiB RAM (49152)` `12 cores` `64G rootfs` on `RaidZ1-6TB` `onboot 1` `vmbr0` `192.168.1.13/24` gw `192.168.1.1`
- Host ROCm upgrade prompt before LXC (checks `get_host_rocm_version` vs `ROCM_VERSION` `10.0.0` default never pinned, switches `stable.repo.amd.com` for `10.x`)
- Prompt-before-redeploy guard, GPU passthrough `226:1/226:129/511:0` (`gfx1150` only)

## GPU Passthrough

- Same as `hlh-ai-engine` `112`: `/dev/dri/card0` + `/dev/dri/renderD128` + `/dev/kfd` (`gfx1150` Strix Halo, `c9:00.0` `150e` APU); K80 DRM nodes intentionally excluded

## ROCm / Runtime (0.2.0 — docker image based; validation pending, see 10_ACTIVE.md)

- Host ROCm `10.0.0` default never pinned (provides `amdgpu` kernel driver/firmware behind `/dev/kfd`), `ROCM_VERSION=7.14.1` still via env
- vLLM runs as docker image `vllm/vllm-openai-rocm:latest` (official ROCm build; no in-LXC ROCm install, no `/opt/vllm-venv`, no wheel shims — the venv path and its patches are fully removed)
- `HSA_OVERRIDE_GFX_VERSION=11.0.0` set as container env (default in `/etc/vllm.env`)
- Docker installed in-LXC via `get.docker.com` (fallback `docker.io`)

## Services (0.2.0)

- `vllm.service` `systemd` `EnvironmentFile=-/etc/vllm.env` `ExecStartPre=docker rm -f vllm` `ExecStart=/usr/local/bin/vllm-docker-run.sh` (runner pulls `VLLM_IMAGE` from `/etc/vllm.env` at start) (docker run `--network host` `--ipc host` `--shm-size 16g` `--device /dev/kfd /dev/dri/renderD128 /dev/dri/card0` numeric `--group-add` `--security-opt apparmor=unconfined seccomp=unconfined` model dir ro-mounted at `/srv/ai/models`; auto-prepends `serve` if the image has no entrypoint) `Restart=on-failure`
- `open-webui.service` — **removed from bootstrap in 0.2.0** (phase 10)
- `vllm-switch-model.sh` (`/usr/local/bin` + `/srv/ai/models/vllm-switch-model.sh`) edits `/etc/vllm.env` (`VLLM_MODEL_PATH`/`VLLM_SERVED_NAME`, local dirs or HF ids), restarts, health-probes `/health` up to 180 s

## Model Management (0.2.0)

- Default `/srv/ai/models/Qwen3.5-9B` (`qwen3.5-9b`, 18 GB bf16 safetensors, `qwen3_5` VLM), `--gpu-memory-utilization 0.40` `--max-model-len 4096` `--enforce-eager` `--trust-remote-code` `--limit-mm-per-prompt '{"image":1,"video":1}'` `--mm-processor-cache-gb 1`
- Shared `/srv/ai/models` bind mount with siblings (`112` GGUF/llama.cpp, `113` safetensors); HF cache at `/srv/ai/models/.hf-cache`
- Co-tenancy policy: sibling `112` (primary llama.cpp engine) is **never stopped** by this repo; `VLLM_GPU_MEM_UTIL=0.40` chosen for coexistence (lower to `0.30` if OOM)

## Networking

- vLLM API `http://192.168.1.13:8000` (`/health`, `/v1/models`, `/v1/chat/completions`) — OpenAI-compatible
- Open WebUI — phase 10 (not configured in 0.2.0)

## Bash IaC

- Pure bash `deploy-hlh-ai-engine-vllm.sh` + `configure-hlh-ai-engine-vllm.sh` + `configure-ai-engine-inside-lxc.sh` (pct push/exec)
- LXC `113` `192.168.1.13/24` privileged `nesting,keyctl` `RaidZ1-6TB`

## Verification

- `docker ps` `systemctl status vllm` `curl :8000/health` `curl :8000/v1/models` `docker logs vllm` (host `rocm-smi --showmeminfo vram` + LXC `free -g` for memory gauges)
