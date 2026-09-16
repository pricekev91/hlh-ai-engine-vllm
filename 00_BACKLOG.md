# BACKLOG

Items for future implementation. These are human-entered ideas not yet reflected in the codebase.

## GPU / ROCm

- Track ROCm version compatibility matrix across Proxmox kernel updates (10.0.0 default never pinned, host must match LXC)
- Add GPU memory utilization monitoring for vLLM (`gpu_memory_utilization` vs `rocm-smi` VRAM%)

## Model Management

- Add HF model prefetch/cache warming for vLLM (warm `Qwen/Qwen2.5-Coder-32B` on bootstrap vs lazy download)
- Add model versioning for HF safetensors (pin commit SHAs)

## LXC Lifecycle

- Add LXC snapshot before major vLLM version bumps
- Add `hlh-ai-engine` (112) vs `hlh-ai-engine-vllm` (113) mutual-exclusion guard (same `gfx1150`)

## Open WebUI (phase 10 — deliberately removed from phase-1 bootstrap in 0.2.0)

- Re-add `open-webui.service` on `192.168.1.13:80` (docker `ghcr.io/open-webui/open-webui:main`, `--network host`, `--security-opt apparmor=unconfined`, `OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1`, `BYPASS_MODEL_ACCESS_CONTROL=true`)
- Add Open WebUI auth hardening (default allows all, `BYPASS_MODEL_ACCESS_CONTROL=true`)
- Add HTTPS/TLS termination for `:8080` (currently http only)
- Add `WEBUI_SECRET_KEY` persistence across redeploys

## Docker storage

- Give LXC 113 a dedicated docker data-root subvolume (like the `hlh-docker` pattern on the host) — the rocm image is large and the 64G rootfs is shared with the OS

## Ansible

- Add ansible-lint to CI
- Split bootstrap into roles (docker, vllm.env, vllm.service, open-webui)

## OpenTofu

- Migrate telmate/proxmox -> bpg/proxmox (align with hlh-docker)

## Observability

- Add Prometheus metrics for vLLM (`--enable-metrics` + `/metrics`) and Open WebUI logs
- Add vLLM deprecation for `gfx1150` APU (`enforce_eager` tracking)

## Deployment

- Add `--skip-host-driver` flag like `hlh-ai-engine-k80` (skip host ROCm check after first reboot)
- Pin a known-good `vllm/vllm-openai-rocm` tag after phase-1 validation (default is `:latest`, never pinned)
