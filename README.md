# hlh-ai-engine-vllm

Infrastructure-as-Code for the HLH vLLM inference engine. Deploys **native vLLM (no docker)**
as a Proxmox LXC container with AMD GPU passthrough (Radeon 890M, gfx1150/Strix Halo),
serving local safetensors models from the shared `/srv/ai/models` pool.

## Executive Summary

This repository deploys and configures the **engine-vllm** LXC on the HLH Proxmox host `prox01`
(`192.168.1.10`). It is a sibling of `hlh-ai-engine` (llama.cpp `112` / `192.168.1.12`) and
`hlh-ai-engine-k80` (CUDA `131`), running the shared AI workload on the high-throughput vLLM engine.

**Phase 2 (current): native vLLM, no docker.** Open WebUI is deliberately **not** part of the bootstrap
(anything UI is phase 10).

- LXC `113`, hostname `hlh-ai-engine-vllm`, IP `192.168.1.13` (gw `192.168.1.1`)
- vLLM runs **natively** in the LXC (no docker). ROCm userspace installed in-container
  (matching host `ROCM_VERSION`, default `10.0.0`, never pinned). Python venv at `/opt/vllm-venv`
  with vLLM installed via pip with ROCm support. The host ROCm stack still matters:
  it provides the `amdgpu` kernel driver/firmware behind `/dev/kfd`.
- GPU: AMD Radeon 890M iGPU (`gfx1150`, `c9:00.0`) — `/dev/kfd` + `/dev/dri/renderD128` +
  `/dev/dri/card0` passed through; `HSA_OVERRIDE_GFX_VERSION=11.0.0` set in service env
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
- In-container ROCm userspace install + Python venv + `vllm.service` running native vLLM (runtime config in `/etc/vllm.env`)

**Does not own:**
- Proxmox host kernel pin (that is `iac-hlh` / `proxmox-boot-tool`)
- Application logic or dashboard code (that is `TrashPanda`, `BrickCipher`, etc.)
- `hlh-ai-engine` llama.cpp LXC itself (sibling `112`, same GPU — **never stopped by this repo**)
- Open WebUI (phase 10)

## Quick Start

Full (re)deploy from the Proxmox host. If LXC `113` exists you get a prompt:

```bash
./deploy-hlh-ai-engine-vllm.sh
#   y = destroy & recreate from scratch (full rebuild, ~10-20 min, models persist on ZFS)
#   u = update in-place: patch existing LXC (fast, ~2-5 min — use for iter on bootstrap)
#   n = abort

# Non-interactive:
./deploy-hlh-ai-engine-vllm.sh --update   # fast patch path (same as answering 'u')
./deploy-hlh-ai-engine-vllm.sh --destroy  # full rebuild (same as answering 'y')
# Overrides:
ROCM_VERSION=10.0.0 ./deploy-hlh-ai-engine-vllm.sh --update
```

Reconfigure an existing LXC via Ansible (no recreate — this is the phase-2 path for the live box):

```bash
./configure-hlh-ai-engine-vllm.sh
./configure-hlh-ai-engine-vllm.sh --host 192.168.1.13
```

Or manually on the LXC:

```bash
# from repo root, on the Proxmox host:
pct push 113 ansible/files/configure-ai-engine-inside-lxc.sh /root/ai-engine-bootstrap/configure-ai-engine-inside-lxc.sh --perms 0755
pct exec 113 -- env ROCM_VERSION=10.0.0 bash /root/ai-engine-bootstrap/configure-ai-engine-inside-lxc.sh
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
   `ansible/files/configure-ai-engine-inside-lxc.sh` via `pct push` (`ROCM_VERSION` forwarded).
2. **Configuration**: `ansible/playbooks/hlh-ai-engine-vllm.yml` runs
   `ansible/files/configure-ai-engine-inside-lxc.sh` inside the container:
   installs ROCm userspace, Python venv, vLLM via pip, writes `/etc/vllm.env` + `/usr/local/bin/vllm-run.sh` + `vllm.service`,
   starts vLLM, health-probes.

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
| vLLM runtime | Native (Python venv `/opt/vllm-venv`, no docker) |
| Model storage | `/srv/ai/models` host (RaidZ1-6TB) ↔ `/srv/ai/models` LXC (`mp0`, same path as siblings, `775`) |
| Default model | `/srv/ai/models/Qwen3.5-9B` (18 GB bf16 safetensors, `qwen3_5` VLM; served as `qwen3.5-9b`) |
| GPU device | `/dev/kfd` `511:0` + `/dev/dri/renderD128` `226:128` + `/dev/dri/card0` `226:0` (gfx1150 890M only) |
| Host ROCm | `10.0.0` default never pinned (provides amdgpu kernel driver/firmware) |
| In-LXC ROCm | Userspace installed matching host version (default `10.0.0`) |
| Open WebUI | **not configured — phase 10** |

## Repository Layout

```
hlh-ai-engine-vllm/
├── deploy-hlh-ai-engine-vllm.sh          # LXC 113 creation + GPU passthrough + host ROCm check + bootstrap push
├── configure-hlh-ai-engine-vllm.sh       # Ansible-based reconfiguration (existing LXC, no recreate)
├── ansible/
│   ├── inventories/hlh-ai-engine-vllm.yml
│   ├── playbooks/hlh-ai-engine-vllm.yml
│   └── files/configure-ai-engine-inside-lxc.sh  # ROCm + venv + vLLM + /etc/vllm.env + vllm-run.sh + vllm.service :8000
├── opentofu/
│   ├── main.tf
│   └── variables.tf
├── 00_BACKLOG.md
├── 10_ACTIVE.md
├── 90_DONE.md
├── CHANGELOG.md
├── checkpoint.md                          # historical: venv-era ROCm debugging
├── vllm-lemonade.sh                       # one-off Lemonade-based installer (experiment, not the deploy path)
└── README.md
```

## GPU Backend Notes

**vLLM ROCm native (no docker).** The bootstrap installs ROCm userspace in the LXC (matching host version),
creates a Python venv, and installs vLLM via pip with ROCm support. This replaces the earlier
docker-image approach (`vllm/vllm-openai-rocm`) and the venv-era debugging (see `checkpoint.md`).

- `gfx1150` is community-tier for vLLM ROCm (APU `150e`, UMA ~48 GB + GTT ≈ 68.7 GB total pool).
- `HSA_OVERRIDE_GFX_VERSION=11.0.0` (set in service env, default in `/etc/vllm.env`) is
  **mandatory** for the vLLM/torch combo validated on this box. `11.5.0` → `HIP error:
  invalid device function` at first kernel launch. If a future vLLM ships native gfx1150
  kernels, verify and drop the override.
- `VLLM_USE_V2_MODEL_RUNNER=0` is mandatory on ROCm with the PyPI CUDA wheel (no UVA op
  `get_cuda_view_from_cpu_tensor`); V2 runner crashes at `vllm/utils/torch_utils.py:916` → `vllm/v1/worker/gpu/buffer_utils.py:50`
  (see `checkpoint.md` §5.4). Set in `/etc/vllm.env`, `vllm-run.sh`, and `vllm.service`.
- `--enforce-eager` is on by default (APU lacks some flash-attention / graph-capture paths).
- ROCm compatibility patches applied at bootstrap (torch.accelerator shim, SiluAndMul fallback, vllm_c gating with native fallback for `rms_norm`, triton `target_info`/`constexpr_function` shim, amdsmi from `/opt/rocm/share/amd_smi`).
- Service runs as root with GPU devices passed through from host (cgroup2/mount in LXC config).

## vLLM Tuning Reference

Runtime config lives in **`/etc/vllm.env`** (edit → `systemctl restart vllm`, or use
`vllm-switch-model.sh`):

| Var | Default | Description |
|-----|---------|-------------|
| `VLLM_PORT` | `8000` | OpenAI API port |
| `VLLM_MODEL_DIR` | `/srv/ai/models` | Mount source (host↔LXC, also used by vLLM) |
| `VLLM_MODEL_PATH` | `/srv/ai/models/Qwen3.5-9B` | Local dir or HF id |
| `VLLM_SERVED_NAME` | `qwen3.5-9b` | Name exposed in `/v1/models` |
| `VLLM_GPU_MEM_UTIL` | `0.40` | Fraction of the ~68.7 GB GTT pool (weights + KV). Lower to `0.30` if co-tenant 112 needs more |
| `VLLM_MAX_MODEL_LEN` | `4096` | Context window (derived max for Qwen3.5-9B ~40960; 131072 fails validation, see `checkpoint.md`) |
| `HSA_OVERRIDE_GFX_VERSION` | `11.0.0` | gfx1150 kernel target override (see notes) |
| `VLLM_USE_V2_MODEL_RUNNER` | `0` | Disable V2 runner on ROCm CUDA-wheel (mandatory, §notes) |
| `VLLM_LOG_LEVEL` | `INFO` | vLLM logging level |
| `VLLM_EXTRA_ARGS` | *(empty)* | Extra `vllm serve` args, e.g. `--max-num-seqs 8 --skip-mm-profiling` |

Fixed `vllm serve` flags (in `/usr/local/bin/vllm-run.sh`): `--host 0.0.0.0`,
`--enforce-eager`, `--trust-remote-code`, `--limit-mm-per-prompt '{"image":1,"video":1}'`
(qwen3_5 is a VLM), `--mm-processor-cache-gb 1`.

## Health Checks & Service Lifecycle

| Check | Command |
|-------|---------|
| vLLM status | `systemctl status vllm` |
| vLLM health | `curl -s http://127.0.0.1:8000/health` |
| vLLM models | `curl -s http://127.0.0.1:8000/v1/models` |
| GPU (host) | `rocm-smi` on `192.168.1.10` |
| GPU (LXC) | `rocm-smi` inside LXC (after ROCm install) |
| Memory gauges | `rocm-smi --showmeminfo vram` (host) + `free -g` (LXC) |
| Logs vLLM | `journalctl -u vllm -f` |
| Engine root cause | last `Error` line in the `core.py` traceback in journalctl |
| Deployed config | `cat /etc/vllm.env` |

`vllm-switch-model.sh` probes `/health` up to 180 s after restart (18 GB model load from ZFS
takes several minutes).

## Gotchas

- **`HSA_OVERRIDE_GFX_VERSION=11.0.0`** is mandatory with the validated vLLM/torch combo (§notes).
- **`VLLM_USE_V2_MODEL_RUNNER=0`** is mandatory on ROCm with the PyPI CUDA wheel (§notes; V2 needs `torch.ops._C.get_cuda_view_from_cpu_tensor` which is CUDA-only).
- **`VLLM_MAX_MODEL_LEN=4096`** — 131072 fails validation for Qwen3.5-9B (derived max ~40960); 4096 matches `checkpoint.md` proven safe for co-tenancy.
- **Memory**: vLLM claims `VLLM_GPU_MEM_UTIL` × ~68.7 GB GTT at startup (weights + KV pre-alloc).
  `0.40` ≈ 27 GB. Sibling `112` (llama.cpp) shares the same physical RAM — keep an eye on
  `free -g` / host `rocm-smi`. Lower to `0.30` for patch-test cycles.
- **Co-tenancy**: `112` and `113` share `c9:00.0`. This repo never stops `112`.
- **K80s** on this host are not recognized by the driver and are out of scope (see `hlh-ai-engine-k80` / future V100/MI50 shopping).
- Engine restart cycle ≈ 2–5 min (18 GB weights off ZFS + Python import + model load).
- **Deploy prompt:** existing LXC `113` now offers `y` (destroy/recreate) vs `u` (update in-place fast patch) vs `n` (abort). Use `u`/`--update` for quick iter on bootstrap; `y`/`--destroy` only when you need a clean slate.
- ROCm version in LXC must match host major version (handled by deploy script).

## Governance

This repo is a sibling of `hlh-ai-engine` `112` and `hlh-ai-engine-k80` `131`. They share
`/srv/ai/models` (host `RaidZ1-6TB` dataset `ai/models` at `/srv/ai/models`, `zfs xattr,noacl`,
`775`). `112` hosts the primary llama.cpp inference engine and **must not be stopped** for work
on `113`. See HLH Agile Design Handbook.