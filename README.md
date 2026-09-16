# hlh-ai-engine-vllm

Infrastructure-as-Code for the HLH vLLM inference engine. Deploys **vLLM (official
`vllm/vllm-openai-rocm` docker image)** as a Proxmox LXC container with AMD GPU
passthrough (Radeon 890M, gfx1150/Strix Halo), serving local safetensors models
from the shared `/srv/ai/models` pool.

## Executive Summary

This repository deploys and configures the **engine-vllm** LXC on the HLH Proxmox host `prox01`
(`192.168.1.10`). It is a sibling of `hlh-ai-engine` (llama.cpp `112` / `192.168.1.12`) and
`hlh-ai-engine-k80` (CUDA `131`), running the shared AI workload on the high-throughput vLLM engine.

**Phase 1 (current): get vLLM serving.** Open WebUI is deliberately **not** part of the bootstrap
(anything UI is phase 10).

- LXC `113`, hostname `hlh-ai-engine-vllm`, IP `192.168.1.13` (gw `192.168.1.1`)
- vLLM runs as a **docker container** from the official ROCm image `vllm/vllm-openai-rocm:latest`
  (ROCm userspace ships in the image — **no in-LXC ROCm install, no Python venv, no wheel patches**).
  The host ROCm stack (default `10.0.0`, never pinned) still matters: it provides the `amdgpu`
  kernel driver/firmware behind `/dev/kfd`.
- GPU: AMD Radeon 890M iGPU (`gfx1150`, `c9:00.0`) — `/dev/kfd` + `/dev/dri/renderD128` +
  `/dev/dri/card0` passed through; `HSA_OVERRIDE_GFX_VERSION=11.0.0` set inside the container
  (the proven override for this vLLM/torch combo on gfx1150 — `11.5.0` gives
  `HIP error: invalid device function`, see `checkpoint.md`)
- Serves **`/srv/ai/models/Qwen3.5-9B`** (18 GB bf16 safetensors, `qwen3_5` hybrid
  linear-attention VLM) as OpenAI-compatible API `qwen3.5-9b` on port `8000`
- Model storage on `RaidZ1-6TB` ZFS pool: host `/srv/ai/models` → LXC `/srv/ai/models`
  (`mp0` bind, same path as siblings; GGUF files for the llama.cpp sibling are ignored by vLLM)

> **GPU co-tenancy:** LXC `112` (`hlh-ai-engine`, llama.cpp at `192.168.1.12`) runs on the **same**
> 890M iGPU and hosts the primary inference engine — **do not stop it for this deployment**.
> vLLM's `--gpu-memory-utilization` is set conservatively (`0.40` of the ~68.7 GB GTT pool,
> i.e. ~27 GB incl. the 18 GB weights) so the two can coexist. If the box OOMs, lower
> `VLLM_GPU_MEM_UTIL` in `/etc/vllm.env` (e.g. `0.30`).

## Repository Boundary

**Owns:**
- LXC lifecycle (create, configure, start) on Proxmox `prox01` (`113` privileged `nesting,keyctl`, `48G RAM`, `12 cores`, `64G rootfs` on `RaidZ1-6TB`)
- GPU passthrough for ROCm (`/dev/dri/card0` `226:0`, `renderD128` `226:128`, `/dev/kfd` `511:0` — 890M `gfx1150` only; K80 nodes intentionally excluded)
- Model storage mount wiring (`--mp0 /srv/ai/models,mp=/srv/ai/models`)
- In-container docker install + `vllm.service` running `vllm/vllm-openai-rocm` (runtime config in `/etc/vllm.env`)

**Does not own:**
- Proxmox host kernel pin (that is `iac-hlh` / `proxmox-boot-tool`)
- Application logic or dashboard code (that is `TrashPanda`, `BrickCipher`, etc.)
- `hlh-ai-engine` llama.cpp LXC itself (sibling `112`, same GPU — **never stopped by this repo**)
- Open WebUI (phase 10)

## Quick Start

Full (re)deploy from the Proxmox host (destroys/recreates LXC `113` from scratch; models persist via the ZFS bind):

```bash
./deploy-hlh-ai-engine-vllm.sh
# Overrides:
VLLM_IMAGE=vllm/vllm-openai-rocm:latest ROCM_VERSION=10.0.0 ./deploy-hlh-ai-engine-vllm.sh
```

Reconfigure an existing LXC via Ansible (no recreate — this is the phase-1 path for the live box):

```bash
./configure-hlh-ai-engine-vllm.sh
./configure-hlh-ai-engine-vllm.sh --host 192.168.1.13
```

Or manually on the LXC:

```bash
# from repo root, on the Proxmox host:
pct push 113 ansible/files/configure-ai-engine-inside-lxc.sh /root/ai-engine-bootstrap/configure-ai-engine-inside-lxc.sh --perms 0755
pct exec 113 -- env VLLM_IMAGE=vllm/vllm-openai-rocm:latest bash /root/ai-engine-bootstrap/configure-ai-engine-inside-lxc.sh
```

Switch the loaded model (inside the LXC):

```bash
vllm-switch-model.sh          # also at /srv/ai/models/vllm-switch-model.sh
# edits /etc/vllm.env (VLLM_MODEL_PATH + VLLM_SERVED_NAME), restarts, health-probes :8000
curl -s http://127.0.0.1:8000/health; curl -s http://127.0.0.1:8000/v1/models | head -50
```

Chat completion smoke test:

```bash
curl -s http://192.168.1.13:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-9b","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":16}'
```

## Deployment Model

Deployment and configuration are separate phases:

1. **Provisioning**: `deploy-hlh-ai-engine-vllm.sh` creates privileged LXC `113`, wires GPU
   passthrough (`card0` + `renderD128` + `kfd` only — K80 nodes excluded so ROCm never enumerates
   an unsupported device), checks/upgrades the **host** ROCm if the major mismatches
   (`stable.repo.amd.com` for 10.x, `packages-multi-arch` for 7.x), and pushes
   `ansible/files/configure-ai-engine-inside-lxc.sh` via `pct push` (`VLLM_IMAGE` forwarded).
2. **Configuration**: `ansible/playbooks/hlh-ai-engine-vllm.yml` runs
   `ansible/files/configure-ai-engine-inside-lxc.sh` inside the container:
   installs docker, writes `/etc/vllm.env` + `/usr/local/bin/vllm-docker-run.sh` + `vllm.service`,
   pulls the image, starts vLLM, health-probes.

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
| vLLM image | `vllm/vllm-openai-rocm:latest` (override: `VLLM_IMAGE` env at bootstrap) |
| Model storage | `/srv/ai/models` host (RaidZ1-6TB) ↔ `/srv/ai/models` LXC (`mp0`, same path as siblings, `775`) |
| Default model | `/srv/ai/models/Qwen3.5-9B` (18 GB bf16 safetensors, `qwen3_5` VLM; served as `qwen3.5-9b`) |
| GPU device | `/dev/kfd` `511:0` + `/dev/dri/renderD128` `226:128` + `/dev/dri/card0` `226:0` (gfx1150 890M only) |
| Host ROCm | `10.0.0` default never pinned (provides amdgpu kernel driver/firmware; in-container ROCm comes from the docker image) |
| Open WebUI | **not configured — phase 10** |

## Repository Layout

```
hlh-ai-engine-vllm/
├── deploy-hlh-ai-engine-vllm.sh          # LXC 113 creation + GPU passthrough + host ROCm check + bootstrap push
├── configure-hlh-ai-engine-vllm.sh       # Ansible-based reconfiguration (existing LXC, no recreate)
├── ansible/
│   ├── inventories/hlh-ai-engine-vllm.yml
│   ├── playbooks/hlh-ai-engine-vllm.yml
│   └── files/configure-ai-engine-inside-lxc.sh  # docker + /etc/vllm.env + vllm-docker-run.sh + vllm.service :8000
├── opentofu/
│   ├── main.tf
│   └── variables.tf
├── 00_BACKLOG.md
├── 10_ACTIVE.md
├── 90_DONE.md
├── CHANGELOG.md
├── checkpoint.md                          # historical: venv-era ROCm debugging (superseded by docker image)
├── vllm-lemonade.sh                       # one-off Lemonade-based installer (experiment, not the deploy path)
└── README.md
```

## GPU Backend Notes

**vLLM ROCm only, via official docker image.** The `vllm/vllm-openai-rocm` image is a full ROCm
build (torch+HIP+triton inside the image), which replaces the earlier in-LXC venv path
(CUDA wheel + `torch.ops._C` shims — see `checkpoint.md` for that debugging history).

- `gfx1150` is community-tier for vLLM ROCm (APU `150e`, UMA ~48 GB + GTT ≈ 68.7 GB total pool).
- `HSA_OVERRIDE_GFX_VERSION=11.0.0` (set as container env, default in `/etc/vllm.env`) is
  **mandatory** for the vLLM/torch combo validated on this box. `11.5.0` → `HIP error:
  invalid device function` at first kernel launch. If a future image ships native gfx1150
  kernels, verify and drop the override.
- `--enforce-eager` is on by default (APU lacks some flash-attention / graph-capture paths).
- Container flags: `--network host`, `--ipc host`, `--shm-size 16g`, `--device /dev/kfd
  /dev/dri/renderD128 /dev/dri/card0`, numeric `--group-add` (kfd/dri gids),
  `--security-opt apparmor=unconfined` (required in privileged LXC), `--security-opt
  seccomp=unconfined` (KFD ioctl). Model dir mounted read-only at the same path.

## vLLM Tuning Reference

Runtime config lives in **`/etc/vllm.env`** (edit → `systemctl restart vllm`, or use
`vllm-switch-model.sh`):

| Var | Default | Description |
|-----|---------|-------------|
| `VLLM_IMAGE` | `vllm/vllm-openai-rocm:latest` | Container image |
| `VLLM_PORT` | `8000` | OpenAI API port |
| `VLLM_MODEL_DIR` | `/srv/ai/models` | Mount source (host↔LXC, also mounted read-only in container) |
| `VLLM_MODEL_PATH` | `/srv/ai/models/Qwen3.5-9B` | Local dir or HF id |
| `VLLM_SERVED_NAME` | `qwen3.5-9b` | Name exposed in `/v1/models` |
| `VLLM_GPU_MEM_UTIL` | `0.40` | Fraction of the ~68.7 GB GTT pool (weights + KV). Lower to `0.30` if co-tenant 112 needs more |
| `VLLM_MAX_MODEL_LEN` | `4096` | Context window (262k max of the model won't fit in KV) |
| `HSA_OVERRIDE_GFX_VERSION` | `11.0.0` | gfx1150 kernel target override (see notes) |
| `VLLM_SHM_SIZE` | `16g` | Container `/dev/shm` |
| `VLLM_LOG_LEVEL` | `INFO` | vLLM logging level |
| `VLLM_EXTRA_ARGS` | *(empty)* | Extra `vllm serve` args, e.g. `--max-num-seqs 8 --skip-mm-profiling` |

Fixed `vllm serve` flags (in `/usr/local/bin/vllm-docker-run.sh`): `--host 0.0.0.0`,
`--enforce-eager`, `--trust-remote-code`, `--limit-mm-per-prompt '{"image":1,"video":1}'`
(qwen3_5 is a VLM), `--mm-processor-cache-gb 1`.

## Health Checks & Service Lifecycle

| Check | Command |
|-------|---------|
| vLLM status | `systemctl status vllm` |
| vLLM health | `curl -s http://127.0.0.1:8000/health` |
| vLLM models | `curl -s http://127.0.0.1:8000/v1/models` |
| Container | `docker ps` / `docker logs -f vllm` |
| GPU (host) | `rocm-smi` on `192.168.1.10` |
| Memory gauges | `rocm-smi --showmeminfo vram` (host) + `free -g` (LXC) |
| Logs vLLM | `journalctl -u vllm -f` |
| Engine root cause | last `Error` line in the `core.py` traceback of `docker logs vllm` |
| Deployed config | `cat /etc/vllm.env` |

`vllm-switch-model.sh` probes `/health` up to 180 s after restart (18 GB model load from ZFS
takes several minutes).

## Gotchas

- **`HSA_OVERRIDE_GFX_VERSION=11.0.0`** is mandatory with the validated image/torch combo (§notes).
- **Memory**: vLLM claims `VLLM_GPU_MEM_UTIL` × ~68.7 GB GTT at startup (weights + KV pre-alloc).
  `0.40` ≈ 27 GB. Sibling `112` (llama.cpp) shares the same physical RAM — keep an eye on
  `free -g` / host `rocm-smi`.
- **Co-tenancy**: `112` and `113` share `c9:00.0`. This repo never stops `112`.
- **K80s** on this host are not recognized by the driver and are out of scope (see `hlh-ai-engine-k80` / future V100/MI50 shopping).
- Engine restart cycle ≈ 2–5 min (18 GB weights off ZFS + container start).

## Governance

This repo is a sibling of `hlh-ai-engine` `112` and `hlh-ai-engine-k80` `131`. They share
`/srv/ai/models` (host `RaidZ1-6TB` dataset `ai/models` at `/srv/ai/models`, `zfs xattr,noacl`,
`775`). `112` hosts the primary llama.cpp inference engine and **must not be stopped** for work
on `113`. See HLH Agile Design Handbook.
