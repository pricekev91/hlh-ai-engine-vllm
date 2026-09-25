# Backlog — ideas / later work

## CUDA V100 (0.6.0)
- [ ] Post-deploy validation on the V100: tok/s benchmark of Qwen3.6-35B-A3B-GPTQ-Int4 (compare against the 890M era); confirm the selected attention backend in the journal (`Using ... attention backend`)
- [ ] Try removing `--enforce-eager` (CUDA graphs on Volta) — keep it if flaky
- [ ] If both engines (this + llama.cpp on LXC 111) must run concurrently: settle a VRAM split (e.g. `AI_GPU_MEM_UTIL≈0.45` vs llama.cpp context size)
- [ ] Optional: an sm_70 prebuilt flash-attn wheel (vLLM auto-selects; today it falls back to SDPA on Volta)
- [ ] Default `AI_API_KEY` + a firewall rule for :8000 (bound 0.0.0.0)

## General
- [ ] Prometheus scraping of `:8000/metrics` (endpoint always on)
- [ ] Model pool: add more sm_70-friendly checkpoints (fp16 / GPTQ-Int4 / Int8; avoid bf16-only and FP8 — no FP8 on Volta)
- [ ] Optional: move Open WebUI to its own LXC if the shared 48 GB gets tight (currently same LXC as vLLM, no docker)
- [ ] Optional: nightly model-load smoke test (health + 1-token completion)
