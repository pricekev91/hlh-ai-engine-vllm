# Active

## 0.6.0 — GPU backend refactor: ROCm 890M → CUDA V100 eGPU (2026-09-24)
- [x] Research: PyPI `vllm` = CUDA build; vLLM 0.20+ → torch 2.11/cu13 (sm_70 dropped) → pin **0.19.1** (torch 2.10.0+cu128)
- [x] `deploy-hlh-ai-engine-vllm.sh` rewritten: host R580 580.65.06 + V100 validation (driver owned by `hlh-ai-engine-egpu`), LXC 113 creation, `/dev/nvidia*` + UVM passthrough, `--skip-host-driver`/`--update`/`--destroy`
- [x] `configure-hlh-ai-engine-vllm.sh` rewritten: 580 userspace from CUDA ubuntu2404 repo (pinned 580.65.06-0ubuntu1), vLLM 0.19.1 via uv, hard verification (torch cu12.8, single V100, fp16 kernel launch on sm_70, no `+rocm`/`nvidia-*-cu13` packages), native Open WebUI, systemd units, switch script; all ROCm machinery removed
- [x] README (stack-ceiling table, GPU co-tenancy, Volta notes) + CHANGELOG 0.6.0 + checkpoint §10 + tracking files
- [ ] **prox01: pull + `./deploy-hlh-ai-engine-vllm.sh`** (first V100 run; mind co-tenancy with LXC 111)
- [ ] Post-deploy: `/health` + chat via WebUI + `nvidia-smi` VRAM numbers + journal attention-backend line
