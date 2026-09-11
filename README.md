# hlh-ai-engine-vllm

Infrastructure-as-Code for the HLH vLLM inference engine. Deploys a GPU-accelerated
vLLM runtime as a Proxmox LXC container with ROCm support, plus Open WebUI frontend
(vLLM has no built-in web UI).

## Executive Summary

This repository deploys and configures the **engine-vllm** LXC on the HLH Proxmox host `prox01`.
It is a sibling of `hlh-ai-engine` (llama.cpp `112`) and `hlh-ai-engine-k80` (CUDA `131`)
running the same shared AI workload via the high-throughput vLLM engine + Open WebUI chat UI
on discrete AMD hardware (same 890M `gfx1150` — cannot run concurrently with `hlh-ai-engine` due to 48G VRAM contention).

- LXC 113, hostname `hlh-ai-engine-vllm`, IP `192.168.1.13` (gw `192.168.1.1`)
- ROCm `10.0.0` default (2026-08-26 latest; unpinned — override: `ROCM_VERSION=7.14.1 ./deploy-hlh-ai-engine-vllm.sh`) with AMD RDNA 3 890M iGPU (gfx1150, Strix Halo) — deploy always prints version, never pinned
- `vLLM` ROCm (`/opt/vllm-venv`, `HSA_OVERRIDE_GFX_VERSION=11.5.0`, `gpu_memory_utilization=0.85`) serving OpenAI-compatible API on port `8000` (`/health` + `/v1` + `/v1/chat/completions`)
- `Open WebUI` (docker `ghcr.io/open-webui/open-webui:main`) on port `8080` (`OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1`), `--network host` inside same LXC
- Model storage on `RaidZ1-6TB` ZFS pool (`/srv/ai/models` host → `/srv/ai/models` LXC bind mount, same path as siblings; HF cache at `/srv/ai/models/.hf-cache`, GGUF for llama sibling ignored by vLLM)

> **Memory caveat:** 890M is 48G VRAM (UMA 88G with GTT). vLLM claims `gpu_memory_utilization` fraction of total VRAM for KV cache at startup (default `0.85`). `hlh-ai-engine` (`112`) and this `113` share the single iGPU — stop the other first: `pct stop 112 && pct start 113` (or vice versa). `hlh-ai-engine-k80` (`131`) uses separate OCuLink Tesla, so it can run alongside either.

## Repository Boundary

**Owns:**
- LXC lifecycle (create, configure, start) on Proxmox `prox01` (`113` privileged `nesting,keyctl`, `48G RAM`, `12 cores`, `64G rootfs` on `RaidZ1-6TB`)
- GPU passthrough for ROCm (`/dev/dri/card1` `226:1`, `renderD129` `226:129`, `/dev/kfd` `511:0` — 890M `gfx1150` only, same as `hlh-ai-engine`)
- Model storage mount wiring (`--mp0 /srv/ai/models,mp=/srv/ai/models` `775`)
- In-container ROCm (`amdrocm${ROCM_MM}-gfx1150`) + Python venv `vllm` + docker `open-webui` installation (mirrors `hlh-ai-engine` host-upgrade logic)

**Does not own:**
- Proxmox host kernel pin (that is `iac-hlh` / `proxmox-boot-tool`)
- Application logic or dashboard code (that is `TrashPanda`, `BrickCipher`, etc.)
- `hlh-ai-engine` llama.cpp LXC itself (sibling, same GPU — mutual exclusion)

## Quick Start

Deploy the vLLM engine LXC on the Proxmox host (upgrades host ROCm if needed, then LXC — always prints version, never pinned):

```bash
./deploy-hlh-ai-engine-vllm.sh              # default 10.0.0 (latest 2026-08-26) — prompts to upgrade host 7.14→10.0 if needed
# Override ROCm version (never pinned):
ROCM_VERSION=7.14.1 ./deploy-hlh-ai-engine-vllm.sh   # stay on older stable to match host without upgrade
# Bootstrap also respects: ROCM_VERSION=10.0.0 bash ansible/files/configure-ai-engine-inside-lxc.sh
# Host must match LXC major (7.x vs 10.x): deploy checks host and prompts via stable.repo.amd.com / packages-multi-arch
```

Reconfigure an existing LXC via Ansible (no recreate):

```bash
./configure-hlh-ai-engine-vllm.sh
./configure-hlh-ai-engine-vllm.sh --host 192.168.1.13
```

Switch loaded vLLM model (inside LXC after deployment):

```bash
vllm-switch-model.sh              # interactive: HF model ID + served name, rewrites vllm.service
# also at /srv/ai/models/vllm-switch-model.sh
curl -s http://127.0.0.1:8000/health; curl -s http://127.0.0.1:8000/v1/models | head -50
```

Open WebUI chat:

```
http://192.168.1.13:8080   # login admin on first visit, models pulled from vLLM :8000
```

## Deployment Model

Deployment and configuration are separate phases (same as `hlh-ai-engine`):

1. **Provisioning**: `deploy-hlh-ai-engine-vllm.sh` creates privileged LXC `113`, wires GPU passthrough
   (`card1`+`renderD129`+`kfd` only — `gfx803` excluded), prints `ROCm ${ROCM_VERSION}` + `vLLM ROCm gfx1150 + Open WebUI`,
   upgrades host ROCm if major mismatch (`7.14`→`10.0` via `stable.repo.amd.com` prompt), and pushes `ansible/files/configure-ai-engine-inside-lxc.sh` via `pct push` (`env ROCM_VERSION=...` forwarded).
2. **Configuration**: `ansible/playbooks/hlh-ai-engine-vllm.yml` runs `ansible/files/configure-ai-engine-inside-lxc.sh` inside the container
   (installs `ROCM_VERSION` `amdrocm${MM}-gfx1150`, creates `/opt/vllm-venv`, `pip install vllm[rocm]`, installs `docker.io`, creates `vllm.service` `:8000` + `open-webui.service` `:8080` docker with `--network host`).

## OpenTofu Module

For programmatic LXC creation via OpenTofu (bind mount, not storage volume):

```hcl
module "hlh_ai_engine_vllm" {
  source = "./opentofu"
  pm_api_url          = var.pm_api_url
  pm_api_token_id     = var.pm_api_token_id
  pm_api_token_secret = var.pm_api_token_secret
  target_node         = "prox01"
  hostname            = "hlh-ai-engine-vllm"
  vmid                = 113  # .13 parity with 192.168.1.13
  ip_cidr             = "192.168.1.13/24"
  # ... other variables (see opentofu/variables.tf)
}
# GPU cgroup/mount for /dev/dri + /dev/kfd appended by deploy-hlh-ai-engine-vllm.sh post-create
```

## Runtime Contract

| Item | Value |
|------|-------|
| LXC | `113` `hlh-ai-engine-vllm` `192.168.1.13/24` `prox01` `RaidZ1-6TB` |
| vLLM API (OpenAI) | `http://192.168.1.13:8000` (`/health`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`) |
| Open WebUI | `http://192.168.1.13:8080` (docker `open-webui`, `OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1`) |
| Model storage | `/srv/ai/models` host (RaidZ1-6TB) ↔ `/srv/ai/models` LXC (`mp0`, same path as siblings, `775`); HF cache `/.hf-cache` |
| GPU device | `/dev/dri/card1` `226:1` + `renderD129` `226:129` + `/dev/kfd` `511:0` (`gfx1150` 890M only) |
| Default model | `Qwen/Qwen2.5-Coder-32B-Instruct` (HF safetensors, `served-model-name qwen2.5-coder-32b`, `gpu_memory_utilization 0.85`, `max_model_len 8192`, `enforce_eager`) |
| ROCm | `10.0.0` default never pinned (`amdrocm10.0-gfx1150` or generic `amdrocm10.0` via `ROCM_MM`) |

## Repository Layout

```
hlh-ai-engine-vllm/
├── deploy-hlh-ai-engine-vllm.sh          # LXC 113 creation + GPU passthrough + host ROCm upgrade + bootstrap
├── configure-hlh-ai-engine-vllm.sh       # Ansible-based reconfiguration
├── ansible/
│   ├── inventories/hlh-ai-engine-vllm.yml
│   ├── playbooks/hlh-ai-engine-vllm.yml
│   └── files/configure-ai-engine-inside-lxc.sh  # vLLM venv + vllm.service :8000 + open-webui.service :8080
├── opentofu/
│   ├── main.tf
│   └── variables.tf
├── 00_BACKLOG.md
├── 10_ACTIVE.md
├── 90_DONE.md
├── CHANGELOG.md
└── README.md
```

## GPU Backend Notes

**vLLM ROCm only, Open WebUI is UI only.** `vLLM` has no built-in WebUI (unlike `llama.cpp` `:80` and `ollama`). Correct split is `vLLM :8000` (inference, `HSA_OVERRIDE_GFX_VERSION=11.5.0`, `gfx1150` via `amdrocm${MM}-gfx1150`) + `Open WebUI :8080` (chat, `ghcr.io/open-webui/open-webui:main` docker `--network host`).

- ROCm `10.0.0` default never pinned (host must match LXC major; deploy prompts `7.14→10.0` upgrade via `stable.repo.amd.com` for `debian13`/`ubuntu2404`). `7.14.1` still via `ROCM_VERSION=7.14.1`.
- `gfx1150` is community-tier for `vLLM` ROCm (APU `150e` UMA `48G VRAM + 40G GTT = 88G`); `vLLM` uses `gpu_memory_utilization` (default `0.85`) to carve KV cache at startup. `hlh-ai-engine` `llama.cpp` `HIP+Vulkan` handles large `96K` contexts via Vulkan GTT; `vLLM` may need `--enforce-eager` for APU.

## vLLM Tuning Reference

Default `vllm.service` flags (from systemd unit):

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-Coder-32B-Instruct` | HF model ID |
| `--host` | `0.0.0.0` | Listen all |
| `--port` | `8000` | OpenAI API |
| `--served-model-name` | `qwen2.5-coder-32b` | Name exposed in `/v1/models` |
| `--gpu-memory-utilization` | `0.85` | Fraction of VRAM for KV cache |
| `--max-model-len` | `8192` | Context window |
| `--enforce-eager` | `true` | Disable CUDA graphs (needed for APU) |
| `--dtype` | `half` | `fp16` |

Open WebUI env (from `open-webui.service`):

| Var | Value |
|-----|-------|
| `PORT` | `8080` |
| `OPENAI_API_BASE_URL` | `http://127.0.0.1:8000/v1` |
| `BYPASS_MODEL_ACCESS_CONTROL` | `true` |

Switch model: `vllm-switch-model.sh` rewrites `--model` + `--served-model-name` and `systemctl restart vllm` (health probes `http://127.0.0.1:8000/health`).

## Health Checks & Service Lifecycle

| Check | Command |
|-------|---------|
| vLLM status | `systemctl status vllm` |
| WebUI status | `systemctl status open-webui` / `docker ps` |
| vLLM health | `curl -s http://127.0.0.1:8000/health` |
| vLLM models | `curl -s http://127.0.0.1:8000/v1/models` |
| Open WebUI | `curl -s http://127.0.0.1:8080/` |
| GPU HIP | `rocm-smi && hipconfig --version` |
| Logs vLLM | `journalctl -u vllm -f` |
| Logs WebUI | `journalctl -u open-webui -f` / `docker logs -f open-webui` |
| Deployed version | `grep ROCM_VERSION /root/ai-engine-bootstrap/configure-ai-engine-inside-lxc.sh` |

`vllm-switch-model.sh` probes `/health` up to 90s (same as `hlh-ai-engine` `switch-model.sh`). Deploy prints `ROCm version : ${ROCM_VERSION} | Backend: vLLM ROCm + Open WebUI` on `[6/6]`.

## Governance

This repo is a sibling of `hlh-ai-engine` `112` and `hlh-ai-engine-k80` `131`. They share `/srv/ai/models` (host `RaidZ1-6TB` dataset `ai/models` at `/srv/ai/models` `zfs xattr,noacl` `775`). Do not run `112` and `113` concurrently on the same `c9:00.0` `gfx1150` — `pct stop 112` before `pct start 113`. See HLH Agile Design Handbook.
