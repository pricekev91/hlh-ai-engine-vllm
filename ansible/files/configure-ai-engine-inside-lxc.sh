#!/usr/bin/env bash
# configure-ai-engine-inside-lxc.sh (vLLM variant)
# Version: 0.2.0
# Description: Bootstrap vLLM (docker image vllm/vllm-openai-rocm) on Ubuntu 24.04 LXC
#              with ROCm passthrough (gfx1150). PHASE 1: get vLLM serving.
# Target GPU: AMD Radeon 890M (gfx1150/Strix Halo) on Proxmox 9.x privileged LXC — mirrors hlh-ai-engine 113/192.168.1.13
# Requirements: Run as root inside privileged LXC with GPU passthrough (/dev/dri/card0, renderD128, /dev/kfd) and /srv/ai/models bind mount
# Changelog:
#   0.2.0 - PHASE 1 refactor: vLLM now runs as docker image vllm/vllm-openai-rocm (official ROCm build — no in-LXC
#           ROCm install, no /opt/vllm-venv, no pip wheel shims/patches). Runtime config in /etc/vllm.env,
#           container launched by /usr/local/bin/vllm-docker-run.sh under vllm.service.
#           Default model /srv/ai/models/Qwen3.5-9B (qwen3.5-9b, 18GB bf16 safetensors, qwen3_5 VLM).
#           HSA_OVERRIDE_GFX_VERSION default back to 11.0.0 (11.5.0 gave 'HIP error: invalid device function'
#           on this vLLM/torch combo — see checkpoint.md). Open WebUI removed from bootstrap (phase 10).
#   0.1.6 - Fix HSA_OVERRIDE_GFX_VERSION to 11.5.0 (proper gfx1150 for ROCm 10.x), fix host ROCm detection in deploy script to find 10.0.0 (was returning 7.14.0 or unknown)
#   0.1.4 - Fix torch ROCm overwrite (vLLM 0.29.0 pulls CUDA 2.13.0, reinstall 2.8.0+rocm6.4 after vLLM), verify hip
#   0.1.3 - Fix Open WebUI privileged LXC AppArmor (add --security-opt apparmor=unconfined), move WEBUI_PORT 8080 -> 80, default model /srv/ai/models/Qwen3.5-9B-safetensors (qwen3.5-9b, 0.70/4096+mm), mirror live 113 flags
#   0.1.2 - Fix vLLM bootstrap: HF Qwen2.5-7B (not GGUF qwen35moe which 0.6.6 cannot load), GFX 11.0.0 override (11.5.0 gives HIP invalid device), ld.so.conf for libamd_smi, amdsmi 27.0.0
#   0.1.1 - Default model now shared GGUF /srv/ai/models/Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf (same as hlh-ai-engine 112, 98304 ctx) — added hf-cache check, VLLM_LOGGING_LEVEL=DEBUG, pre-check rocm-smi (reverted: GGUF not supported by vLLM 0.6.6)
#   0.1.0 - Initial vLLM variant forked from hlh-ai-engine v0.9.4
#           ROCm 10.0.0 default never pinned (ROCM_VERSION env, 7.14.1 rollback supported)
#           Replaces llama.cpp HIP+Vulkan dual build with vLLM ROCm (pip) + Open WebUI (docker)
#           Model storage /srv/ai/models shared with siblings (GGUF for llama.cpp, HF safetensors for vLLM)

set -euo pipefail

# --- CONFIGURABLE (env-overridable, e.g. pushed via `env VLLM_IMAGE=... pct exec ...`) ---
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai-rocm:latest}"
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL_DIR="${MODEL_DIR:-/srv/ai/models}"
# Default model: GPTQ Int4 quantized (35B MoE, ~20GB)
DEFAULT_MODEL_PATH="${DEFAULT_MODEL_PATH:-${MODEL_DIR}/Qwen3.6-35B-A3B-GPTQ-Int4}"
DEFAULT_MODEL_NAME="${DEFAULT_MODEL_NAME:-qwen3.6-35b-a3b-gptq-int4}"
# gfx1150: 11.0.0 (gfx1100 triton-JIT target) is the proven override for this vLLM/torch combo.
# 11.5.0 -> 'HIP error: invalid device function' at first kernel launch (see checkpoint.md §4.5).
GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
# 0.40 x ~68.7GB GTT pool ~= 27.5GB (18GB weights + KV). Conservative on purpose:
# sibling LXC 112 (hlh-ai-engine, llama.cpp) runs on the SAME iGPU and must not be stopped.
# Lower to 0.30 if the box OOMs under co-tenancy.
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.40}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_SERVICE="/etc/systemd/system/vllm.service"
RUNNER="/usr/local/bin/vllm-docker-run.sh"
VLLM_ENV="/etc/vllm.env"
SWITCH_SCRIPT="/usr/local/bin/vllm-switch-model.sh"

# --- 1. BASE DEPENDENCIES (docker only — ROCm userspace ships inside the image) ---
echo "[1/6] Installing base dependencies (docker, no in-LXC ROCm)..."
apt-get update
apt-get install -y --no-install-recommends \
  curl ca-certificates gnupg openssh-server

if ! command -v docker >/dev/null 2>&1; then
  echo "[1/6] Installing docker via get.docker.com (fallback: docker.io)..."
  curl -fsSL https://get.docker.com | sh 2>&1 | tail -5 \
    || { apt-get install -y --no-install-recommends docker.io; }
fi
systemctl enable docker 2>&1 || true
systemctl start docker 2>&1 || true
docker --version 2>&1 | head -1 || { echo "ERROR: docker not ready" >&2; exit 1; }

# SSH
mkdir -p /etc/ssh/sshd_config.d
cat > /etc/ssh/sshd_config.d/99-root-login.conf <<'EOF'
PermitRootLogin yes
PasswordAuthentication yes
KbdInteractiveAuthentication no
UsePAM yes
EOF
systemctl enable ssh 2>&1 || true
systemctl restart ssh 2>&1 || systemctl restart sshd 2>&1 || true

# --- 2. MODEL DIR ---
echo "[2/6] Checking model directory ${MODEL_DIR}..."
mkdir -p "${MODEL_DIR}"
mkdir -p "${MODEL_DIR}/.hf-cache" 2>&1 || true
if [ -d "${DEFAULT_MODEL_PATH}" ]; then
  echo "Default model present: ${DEFAULT_MODEL_PATH} ($(du -sh "${DEFAULT_MODEL_PATH}" 2>/dev/null | awk '{print $1}'))"
  ls -lh "${DEFAULT_MODEL_PATH}" 2>&1 | head -12 || true
else
  echo "WARNING: Default model ${DEFAULT_MODEL_PATH} not found — vLLM will fail to start until the model is present."
  echo "Available entries in ${MODEL_DIR}:"
  ls -lh "${MODEL_DIR}" 2>&1 | head -30 || true
fi

# GPU device nodes must be passed through (deploy script appends the cgroup/mount entries)
echo "[2/6] Pre-check: /dev/kfd + /dev/dri"
ls -l /dev/kfd /dev/dri/card0 /dev/dri/renderD128 2>&1 | head -10 || true
[[ -e /dev/kfd ]] || echo "WARNING: /dev/kfd missing — GPU passthrough not configured (re-run deploy script [3/6])."

# --- 3. RUNTIME CONFIG (/etc/vllm.env) ---
echo "[3/6] Writing runtime config ${VLLM_ENV}..."
cat > "${VLLM_ENV}" << EOF
# vLLM runtime config (phase 1). Edit, then: systemctl restart vllm
# (or use /usr/local/bin/vllm-switch-model.sh).
VLLM_IMAGE=${VLLM_IMAGE}
VLLM_PORT=${VLLM_PORT}
VLLM_MODEL_DIR=${MODEL_DIR}
VLLM_MODEL_PATH=${DEFAULT_MODEL_PATH}
VLLM_SERVED_NAME=${DEFAULT_MODEL_NAME}
VLLM_GPU_MEM_UTIL=${GPU_MEM_UTIL}
VLLM_MAX_MODEL_LEN=${MAX_MODEL_LEN}
HSA_OVERRIDE_GFX_VERSION=${GFX_VERSION}
VLLM_SHM_SIZE=16g
VLLM_LOG_LEVEL=INFO
# Extra vllm serve args (space-separated, no quoting). Examples:
#   VLLM_EXTRA_ARGS=--max-num-seqs 8 --skip-mm-profiling
VLLM_EXTRA_ARGS=
EOF
chmod 644 "${VLLM_ENV}"

# --- 4. CONTAINER RUNNER + SYSTEMD UNIT ---
echo "[4/6] Writing ${RUNNER} and ${VLLM_SERVICE}..."
cat > "${RUNNER}" << 'RUNNER'
#!/usr/bin/env bash
# vllm-docker-run.sh — ExecStart for vllm.service: launches the vLLM ROCm container.
# /etc/vllm.env is the single source of truth (image, model, flags).
set -euo pipefail
set -a; . /etc/vllm.env; set +a

# Ensure the image is present (fast no-op when already cached).
docker pull "${VLLM_IMAGE}" 2>&1 | tail -2 || {
  echo "ERROR: cannot pull ${VLLM_IMAGE} and no local copy — refusing to start a stale container" >&2
  exit 1
}

# Device nodes passed through to this LXC (890M gfx1150 only — see deploy script GPU section).
KFD_GID="$(stat -c %g /dev/kfd 2>/dev/null || echo 44)"
DRI_GID="$(stat -c %g /dev/dri/renderD128 2>/dev/null || echo 44)"

# Official vLLM images ship ENTRYPOINT ["vllm","serve"]; if this image has none,
# we must pass `serve` ourselves.
EP="$(docker inspect --format '{{json .Config.Entrypoint}}' "${VLLM_IMAGE}" 2>/dev/null || echo null)"
if [[ "${EP}" == "null" || "${EP}" == "[]" || -z "${EP}" ]]; then
  ARGS=(serve)
else
  ARGS=()
fi

ARGS+=(
  "${VLLM_MODEL_PATH}"
  --host 0.0.0.0
  --port "${VLLM_PORT}"
  --served-model-name "${VLLM_SERVED_NAME}"
  --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}"
  --max-model-len "${VLLM_MAX_MODEL_LEN}"
  --enforce-eager
  --trust-remote-code
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --limit-mm-per-prompt '{"image":1,"video":1}'
  --mm-processor-cache-gb 1
)
if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA <<< "${VLLM_EXTRA_ARGS}"
  ARGS+=("${EXTRA[@]}")
fi

# --security-opt apparmor=unconfined : required inside privileged LXC (docker-default fails, exit 125)
# --security-opt seccomp=unconfined  : KFD ioctl can be blocked by default seccomp profile
# --group-add <gid>                 : /dev/kfd + /dev/dri are root:render on host; numeric GIDs always work
exec docker run --rm --name vllm \
  --network host \
  --security-opt apparmor=unconfined \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri/renderD128 \
  --device /dev/dri/card0 \
  --group-add "${KFD_GID}" \
  --group-add "${DRI_GID}" \
  --ipc host \
  --shm-size "${VLLM_SHM_SIZE}" \
  -v "${VLLM_MODEL_DIR}:/srv/ai/models:rw" \
  -e HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION}" \
  -e VLLM_LOGGING_LEVEL="${VLLM_LOG_LEVEL}" \
  -e HF_HUB_CACHE=/srv/ai/models/.hf-cache \
  "${VLLM_IMAGE}" \
  "${ARGS[@]}"
RUNNER
chmod 755 "${RUNNER}"

cat > "${VLLM_SERVICE}" << UNIT
[Unit]
Description=vLLM OpenAI-compatible server (docker vllm/vllm-openai-rocm, image from /etc/vllm.env)
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=simple
# image/model/flags come from /etc/vllm.env at runtime (the runner pulls if needed)
EnvironmentFile=-/etc/vllm.env
ExecStartPre=/usr/bin/docker rm -f vllm
ExecStart=${RUNNER}
ExecStop=/usr/bin/docker stop vllm
TimeoutStopSec=60
Restart=on-failure
RestartSec=10
User=root

[Install]
WantedBy=multi-user.target
UNIT

# --- 5. MODEL SWITCH HELPER ---
cat > "${SWITCH_SCRIPT}" << 'EOS'
#!/usr/bin/env bash
# vllm-switch-model.sh — switch the vLLM model (edits /etc/vllm.env, restarts container)
set -euo pipefail
ENV_FILE=/etc/vllm.env
MODEL_DIR=/srv/ai/models

echo "Current model: $(grep '^VLLM_MODEL_PATH=' "$ENV_FILE" | cut -d= -f2)"
echo ""
echo "Model dirs in ${MODEL_DIR}:"
ls -1 "${MODEL_DIR}" 2>/dev/null | grep -vE '^\.(hf-cache|cache)|^dl.*\.sh$|^switch|\.gguf$|\.bak' | sed 's/^/  /' || true
echo ""
read -rp "Model path (dir in ${MODEL_DIR}, or full path, or HF id): " NEW_MODEL
if [[ -z "${NEW_MODEL}" ]]; then echo "Aborted."; exit 1; fi
if [[ "${NEW_MODEL}" != /* && ! "${NEW_MODEL}" == */* && -d "${MODEL_DIR}/${NEW_MODEL}" ]]; then
  NEW_MODEL="${MODEL_DIR}/${NEW_MODEL}"
fi
if [[ -d "${NEW_MODEL}" ]]; then
  echo "Using local model dir: ${NEW_MODEL}"
else
  echo "Not a local dir — treating as HuggingFace id: ${NEW_MODEL}"
fi
read -rp "Served name [default: $(basename "${NEW_MODEL}" | tr '[:upper:]' '[:lower:]')]: " SERVED
SERVED="${SERVED:-$(basename "${NEW_MODEL}" | tr '[:upper:]' '[:lower:]')}"

sed -i -E "s|^VLLM_MODEL_PATH=.*|VLLM_MODEL_PATH=${NEW_MODEL}|" "$ENV_FILE"
sed -i -E "s|^VLLM_SERVED_NAME=.*|VLLM_SERVED_NAME=${SERVED}|" "$ENV_FILE"
echo "Updated:"
grep -E '^(VLLM_MODEL_PATH|VLLM_SERVED_NAME)=' "$ENV_FILE"

systemctl restart vllm
echo "Waiting for vLLM health on :8000 (model load from ZFS can take minutes)..."
for i in $(seq 1 90); do
  if curl -fsS -m 3 -o /dev/null http://127.0.0.1:8000/health 2>/dev/null; then
    echo "[OK] vLLM serving ${NEW_MODEL} as ${SERVED}"
    curl -s http://127.0.0.1:8000/v1/models | head -100 || true
    exit 0
  fi
  sleep 2
done
echo "WARNING: vLLM did not become healthy — check: journalctl -u vllm -f && docker logs vllm"
EOS
chmod +x "${SWITCH_SCRIPT}"
cp "${SWITCH_SCRIPT}" "${MODEL_DIR}/vllm-switch-model.sh" 2>&1 || true

# --- 6. PULL + ENABLE + START ---
echo "[6/6] Pulling ${VLLM_IMAGE} (large image, may take a while)..."
docker pull "${VLLM_IMAGE}" 2>&1 | tail -5
docker image ls "${VLLM_IMAGE%%:*}" 2>&1 | head -5 || true

systemctl daemon-reload
systemctl enable --now vllm 2>&1 || true

# --- 7. VERIFICATION ---
echo "[7/7] Verifying..."
echo "[docker ps]"
docker ps 2>&1 | head -10 || true
echo ""
echo "[vLLM health (waiting up to 5 min for model load) ...]"
HEALTHY=0
for i in $(seq 1 60); do
  if curl -fsS -m 3 -o /dev/null http://127.0.0.1:8000/health 2>/dev/null; then
    HEALTHY=1; break
  fi
  sleep 5
done
if [[ "${HEALTHY}" == "1" ]]; then
  echo "[health] OK"
  echo "[v1/models]"
  curl -s http://127.0.0.1:8000/v1/models | head -50 || true
else
  echo "[health] NOT healthy yet — vLLM may still be loading (18GB off ZFS) or hit a startup error."
  echo "  journalctl -u vllm -f        # service + pull + runner logs"
  echo "  docker logs vllm             # engine logs (look for last 'Error' in core.py traceback)"
fi
echo ""
systemctl status vllm --no-pager 2>&1 | tail -15 || true
echo ""
echo "[Bootstrap complete - vllm 0.2.0 (phase 1: docker vLLM ROCm)]"
echo "  vLLM API (OpenAI) : http://<container-ip>:${VLLM_PORT}/v1  (health http://<container-ip>:${VLLM_PORT}/health)"
echo "  Image             : ${VLLM_IMAGE}"
echo "  Model             : ${DEFAULT_MODEL_PATH} (served as ${DEFAULT_MODEL_NAME})"
echo "  Runtime config    : ${VLLM_ENV}  (edit + systemctl restart vllm)"
echo "  Switch model      : ${SWITCH_SCRIPT}"
echo "  GPU               : gfx1150 (890M) HSA_OVERRIDE_GFX_VERSION=${GFX_VERSION} (inside container)"
echo "  Open WebUI        : NOT configured (phase 10)"
echo "  Logs              : journalctl -u vllm -f  |  docker logs -f vllm"
