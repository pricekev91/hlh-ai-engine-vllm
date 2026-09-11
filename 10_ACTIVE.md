# TODO

Active items in progress. These are the current focus areas.

## Active

- [ ] Validate vLLM ROCm on gfx1150 890M (Strix Halo) — `Qwen/Qwen2.5-Coder-32B` `gpu_memory_utilization 0.85`, `enforce_eager`
- [ ] Validate Open WebUI -> vLLM :8000 connectivity (`OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1`) on `192.168.1.13:8080`
- [ ] Confirm LXC 113/192.168.1.13 does not collide with 112/131 (same `gfx1150` mutual exclusion)

## This Week

- [ ] Run `ROCM_VERSION=10.0.0 ./deploy-hlh-ai-engine-vllm.sh` full cycle (host upgrade prompt, LXC 113 create, vllm + open-webui health)
- [ ] Curl `http://192.168.1.13:8000/health` + `http://192.168.1.13:8000/v1/models` after bootstrap
- [ ] Browse `http://192.168.1.13:8080` Open WebUI, test chat via vLLM

## Done

- [x] Scaffolding v0.1.0 forked from hlh-ai-engine v0.9.4 (113/192.168.1.13, deploy host upgrade, GPU passthrough, /srv/ai/models shared)
