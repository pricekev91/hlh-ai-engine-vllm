# TODO

Active items in progress. These are the current focus areas.

## Phase 1 — get vLLM serving (docker `vllm/vllm-openai-rocm` + `/srv/ai/models/Qwen3.5-9B`)

- [ ] Deploy on LXC 113 (`./configure-hlh-ai-engine-vllm.sh` or full `./deploy-hlh-ai-engine-vllm.sh`)
- [ ] `curl -s http://192.168.1.13:8000/health` → 200; `/v1/models` shows `qwen3.5-9b`
- [ ] Chat completion smoke test (`--max_tokens 16`) returns sane text
- [ ] Memory check under load: host `rocm-smi --showmeminfo vram` + LXC `free -g` with `112` (llama.cpp) running — confirm `VLLM_GPU_MEM_UTIL=0.40` coexists; drop to `0.30` if OOM
- [ ] Confirm `HSA_OVERRIDE_GFX_VERSION=11.0.0` still required with the image (if `HIP error: invalid device function`, the override is wrong; if a native gfx1150 build exists, test without it)
- [ ] If the image has no `vllm serve` entrypoint, verify `/usr/local/bin/vllm-docker-run.sh` entrypoint-detection kicked in (it auto-prepends `serve`)

## Phase 10 — Open WebUI (deferred, do NOT configure in phase 1)

- [ ] Re-add `open-webui.service` (docker `ghcr.io/open-webui/open-webui:main`, `--network host`, `OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1`) — see `00_BACKLOG.md`

## Done (phase 1 refactor)

- [x] Refactor repo to `vllm/vllm-openai-rocm` docker image (no in-LXC ROCm/venv/wheel shims)
- [x] Default model `/srv/ai/models/Qwen3.5-9B` (qwen3.5-9b), runtime config in `/etc/vllm.env`
- [x] `vllm-switch-model.sh` → env-file based
- [x] README/CHANGELOG/tracking docs updated; co-tenancy policy documented (never stop 112)
