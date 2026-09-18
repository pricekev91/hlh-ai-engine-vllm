# BACKLOG

Items for future implementation. These are human-entered ideas not yet reflected in the codebase.

## GPU / ROCm — P5 proper ROCm build (sustainable fix)

- **P5: Replace CUDA-wheel shims with real ROCm build.** Current native pip `vllm 0.29.0` CUDA wheel requires perpetual `torch.ops._C` shims (`get_cuda_view`, `rms_norm`, `silu_and_mul`, `target_info`, etc.) — per review, unsustainable. Options:
  - **[A] Docker:** `vllm/vllm-openai-rocm:latest` (or ROCm 6.4/7.0 tag) — `vllm.service` runs `docker run --device /dev/kfd --device /dev/dri/renderD128 --network host` with `/srv/ai/models` bind; known-good from `0.2.0` before `0.3.0` phase-2 revert. Pros: ships `vllm._rocm_C` compiled, no shim. Cons: docker in privileged LXC, 64G rootfs needs dedicated subvolume (see Docker storage below).
  - **[B] Source build inside LXC:** `git clone vllm 0.29.0; USE_ROCM=1 MAX_JOBS=12 PYTORCH_ROCM_ARCH=gfx1150 pip install --no-build-isolation -e .` — produces `vllm._rocm_C`/`_C` locally with HIP, eliminates `AttributeError: _C has no attribute X` family. Needs `hipcc`, `rocm-dev`, `~30-60 min` on 12c/48G. Set `HSA_OVERRIDE_GFX_VERSION=11.0.0` still required.
- Decision pending after `0.3.3` quick test — if `rms_norm` fix still hits next `_C` op, switch to P5.

## GPU / ROCm (other)

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
