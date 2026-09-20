#!/usr/bin/env bash
# configure-hlh-ai-engine-vllm.sh
# Version: 0.5.0
# Description: Native vLLM + native Open WebUI on Ubuntu 24.04 LXC via wheels.vllm.ai
#              Target: Radeon 890M (gfx1150 / Strix Point). No Docker — tok/s first.
#
# Design rules (v0.5.0 simplicity):
#   1. Single source of truth: uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/<ver>/<variant>
#      Starting point: uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/0.29.0/rocm723 (today's stable)
#      Auto-resolves latest stable at deploy time — zero script changes when 0.30.0 ships.
#   2. No Docker. Both vLLM and Open WebUI run native via uv venvs in the same LXC (shared GPU, shared /srv/ai/models).
#   3. No ROCm apt in LXC — ROCm userspace comes from the pip wheels (torch bundles HIP libs).
#      Host kernel amdgpu driver behind /dev/kfd is the only host requirement.
#   4. Any failed check aborts loudly. No shims/fallbacks silently hiding a broken env.
#   5. Two venvs: /opt/vllm-venv (vLLM) and /opt/open-webui-venv (WebUI) — avoids dep conflicts.

set -euo pipefail

# --- CONFIGURABLE (env-overridable) ---
VLLM_VERSION="${VLLM_VERSION:-}"               # pin, e.g. 0.29.0; else auto from PyPI
VLLM_ROCM_VARIANT="${VLLM_ROCM_VARIANT:-}"     # pin, e.g. rocm723 or 723; else auto
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"       # wheels are cp312 (manylinux_2_39); 3.14 will fail
VENV_DIR="${VENV_DIR:-/opt/vllm-venv}"
WEBUI_VENV_DIR="${WEBUI_VENV_DIR:-/opt/open-webui-venv}"

AI_PORT="${AI_PORT:-8000}"
WEBUI_PORT="${WEBUI_PORT:-80}"
MODEL_DIR="${MODEL_DIR:-/srv/ai/models}"
DEFAULT_MODEL_PATH="${DEFAULT_MODEL_PATH:-${MODEL_DIR}/Qwen3.5-9B}"
DEFAULT_MODEL_NAME="${DEFAULT_MODEL_NAME:-qwen3.5-9b}"
GPU_MEM_UTIL="${AI_GPU_MEM_UTIL:-0.40}"
MAX_MODEL_LEN="${AI_MAX_MODEL_LEN:-4096}"
ENABLE_ROOT_PASSWORD_SSH="${ENABLE_ROOT_PASSWORD_SSH:-1}"

VLLM_SERVICE="/etc/systemd/system/vllm.service"
WEBUI_SERVICE="/etc/systemd/system/open-webui.service"
RUNNER="/usr/local/bin/vllm-run.sh"
WEBUI_RUNNER="/usr/local/bin/open-webui-run.sh"
ENV_FILE="/etc/vllm.env"
WEBUI_ENV_FILE="/etc/open-webui.env"
SWITCH_SCRIPT="/usr/local/bin/vllm-switch-model.sh"

WHEELS_BASE="https://wheels.vllm.ai/rocm"
PYPI_JSON="https://pypi.org/pypi/vllm/json"

# --- helpers: resolve latest stable + variant + python ---
# All helpers are network-resilient: retry with backoff and trust pinned values
# when deploy already resolved them (fresh LXC network may need ~10s after pct start).
curl_retry() {
  local url="$1" tries=6 delay=3
  for i in $(seq 1 $tries); do
    if curl -fsSL --connect-timeout 10 --max-time 20 -o /dev/null "$url" 2>/dev/null; then
      return 0
    fi
    sleep $delay
  done
  return 1
}
curl_fetch() {
  local url="$1" tries=6 delay=3
  for i in $(seq 1 $tries); do
    if curl -fsSL --connect-timeout 10 --max-time 20 "$url" 2>/dev/null; then
      return 0
    fi
    sleep $delay
  done
  return 1
}

resolve_vllm_version() {
  local ver="$1"
  if [[ -n "$ver" ]]; then
    ver="$(echo "$ver" | tr -d '[:space:]')"
    if curl_retry "${WHEELS_BASE}/${ver}/"; then
      echo "$ver"; return 0
    fi
    echo "WARNING: pinned VLLM_VERSION=${ver} probe failed for ${WHEELS_BASE}/${ver}/ — trusting pinned value (deploy already resolved, LXC network may be warming up)" >&2
    echo "$ver"; return 0
  fi
  local latest=""
  latest="$(curl_fetch "${PYPI_JSON}" | python3 -c "import sys,json;print(json.load(sys.stdin)['info']['version'])" 2>/dev/null || true)"
  if [[ -z "$latest" ]]; then
    echo "FATAL: could not fetch latest vLLM version from ${PYPI_JSON} (network down?)" >&2
    exit 1
  fi
  if curl_retry "${WHEELS_BASE}/${latest}/"; then
    echo "$latest"; return 0
  fi
  echo "PyPI latest ${latest} has no wheels, walking releases..." >&2
  local fallback
  fallback="$(curl_fetch "${PYPI_JSON}" | python3 -c "
import json, re
data=json.load(__import__('sys').stdin)
vers=list(data['releases'].keys())
def key(v):
    m=re.match(r'^([0-9.]+)', v)
    if not m: return (-1,)
    return tuple(int(x) for x in m.group(1).split('.'))
for v in sorted(vers, key=key, reverse=True):
    if re.search(r'rc|dev|post|a|b', v): continue
    print(v)
" 2>/dev/null | head -20)"
  while IFS= read -r v; do
    [[ -z "$v" ]] && continue
    if curl_retry "${WHEELS_BASE}/${v}/"; then
      echo "$v"; return 0
    fi
  done <<< "$fallback"
  echo "FATAL: no PyPI release has wheels at ${WHEELS_BASE}/" >&2
  exit 1
}

resolve_rocm_variant() {
  local ver="$1" pin="$2"
  if [[ -n "$pin" ]]; then
    pin="$(echo "$pin" | sed -E 's/^rocm//; s/[^0-9]//g')"
    local p="rocm${pin}"
    if curl_retry "${WHEELS_BASE}/${ver}/${p}/"; then
      echo "$p"; return 0
    fi
    echo "WARNING: pinned VLLM_ROCM_VARIANT=${p} probe failed for ${WHEELS_BASE}/${ver}/${p}/ — trusting pinned value" >&2
    echo "$p"; return 0
  fi
  local listing
  listing="$(curl_fetch "${WHEELS_BASE}/${ver}/" || true)"
  local variants
  variants="$(echo "$listing" | grep -oE 'rocm[0-9]+/' | tr -d '/' | sort -u)"
  if [[ -z "$variants" ]]; then
    echo "FATAL: no rocm variant found at ${WHEELS_BASE}/${ver}/" >&2
    exit 1
  fi
  echo "$variants" | sed -E 's/rocm//' | sort -n | tail -1 | awk '{print "rocm"$1}'
}

resolve_python_version() {
  local ver="$1" variant="$2"
  local listing
  listing="$(curl_fetch "${WHEELS_BASE}/${ver}/${variant}/vllm/" || true)"
  local whl
  whl="$(echo "$listing" | grep -oE 'vllm-[^"]+\.whl' | grep -vE 'rc|\.dev|dev[0-9]' | head -1 || true)"
  if [[ -z "$whl" ]]; then
    whl="$(echo "$listing" | grep -oE 'vllm-[^"]+\.whl' | head -1 || true)"
  fi
  if [[ -z "$whl" ]]; then
    echo "WARNING: could not list vllm wheel at ${WHEELS_BASE}/${ver}/${variant}/vllm/, defaulting to ${PYTHON_VERSION}" >&2
    echo "${PYTHON_VERSION}"; return 0
  fi
  local cp
  cp="$(echo "$whl" | grep -oE 'cp[0-9]+' | head -1 || true)"
  if [[ -n "$cp" ]]; then
    local num="${cp#cp}"
    echo "${num:0:1}.${num:1}"
    return 0
  fi
  echo "${PYTHON_VERSION}"
}

variant_to_dotted() {
  local v="${1#rocm}"
  if [[ ${#v} -eq 3 ]]; then
    echo "${v:0:1}.${v:1:1}.${v:2:1}"
  elif [[ ${#v} -eq 4 ]]; then
    echo "${v:0:2}.${v:2:1}.${v:3:1}"
  else
    echo "$v"
  fi
}

# --- 0. PRE-FLIGHT ---
echo "[0/8] Pre-flight checks..."
if [[ ! -e /dev/kfd ]]; then
  echo "FATAL: /dev/kfd missing. Fix LXC GPU passthrough (/dev/kfd + /dev/dri) first." >&2
  exit 1
fi
ls -l /dev/kfd /dev/dri/ 2>&1 | head -10 || true

systemctl stop vllm 2>/dev/null || true
systemctl stop open-webui 2>/dev/null || true

# --- 1. BASE PACKAGES (do first so curl + network are ready before resolve) ---
echo "[1/8] Installing base packages..."
export DEBIAN_FRONTEND=noninteractive
# Wait for network briefly before apt (fresh LXC may need a few seconds after pct start)
for i in $(seq 1 12); do
  if getent hosts pypi.org >/dev/null 2>&1 || getent hosts wheels.vllm.ai >/dev/null 2>&1 || ping -c1 -W2 1.1.1.1 >/dev/null 2>&1; then
    break
  fi
  echo "  Waiting for network... ($i/12)"
  sleep 3
done
apt-get update
apt-get install -y --no-install-recommends curl ca-certificates git openssh-server \
  libgomp1 libnuma1 libatomic1 libdrm2 python3-dev build-essential

# Resolve versions AFTER network + curl are ready (single source of truth — deploy does same, but we trust pinned)
echo "[1/8] Resolving vLLM + ROCm variant (pinned: VLLM_VERSION=${VLLM_VERSION:-auto} VLLM_ROCM_VARIANT=${VLLM_ROCM_VARIANT:-auto})..."
RESOLVED_VLLM_VERSION="$(resolve_vllm_version "${VLLM_VERSION}")"
RESOLVED_VARIANT="$(resolve_rocm_variant "${RESOLVED_VLLM_VERSION}" "${VLLM_ROCM_VARIANT}")"
RESOLVED_PYTHON="$(resolve_python_version "${RESOLVED_VLLM_VERSION}" "${RESOLVED_VARIANT}")"
# Honor explicit PYTHON_VERSION if pinned and wheels python differs, warn
if [[ "${PYTHON_VERSION}" != "${RESOLVED_PYTHON}" && -n "${PYTHON_VERSION:-}" ]]; then
  echo "WARNING: PYTHON_VERSION=${PYTHON_VERSION} pinned but wheels want ${RESOLVED_PYTHON} — using pinned ${PYTHON_VERSION}" >&2
  RESOLVED_PYTHON="${PYTHON_VERSION}"
fi
if [[ -z "${PYTHON_VERSION:-}" ]]; then
  PYTHON_VERSION="${RESOLVED_PYTHON}"
else
  PYTHON_VERSION="${RESOLVED_PYTHON}"
fi

echo "  Resolved VLLM          : ${RESOLVED_VLLM_VERSION}"
echo "  Resolved ROCm variant  : ${RESOLVED_VARIANT} ($(variant_to_dotted "${RESOLVED_VARIANT}"))"
echo "  Resolved Python        : ${PYTHON_VERSION}"
echo "  Wheels index           : ${WHEELS_BASE}/${RESOLVED_VLLM_VERSION}/${RESOLVED_VARIANT}/"

VLLM_INDEX="${WHEELS_BASE}/${RESOLVED_VLLM_VERSION}/${RESOLVED_VARIANT}/"

if [[ "${ENABLE_ROOT_PASSWORD_SSH}" == "1" ]]; then
  mkdir -p /etc/ssh/sshd_config.d
  cat > /etc/ssh/sshd_config.d/99-root-login.conf <<'EOF'
PermitRootLogin yes
PasswordAuthentication yes
KbdInteractiveAuthentication no
UsePAM yes
EOF
fi
systemctl enable ssh 2>&1 || true
systemctl restart ssh 2>&1 || systemctl restart sshd 2>&1 || true

# --- 2. uv + vLLM VENV ---
echo "[2/8] Installing uv and creating vLLM venv (Python ${PYTHON_VERSION} at ${VENV_DIR})..."
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
fi
export PATH="/usr/local/bin:${PATH}"

rm -rf "${VENV_DIR}"
uv venv --seed --python "${PYTHON_VERSION}" "${VENV_DIR}"
PY="${VENV_DIR}/bin/python"

# --- 3. INSTALL vLLM (simple one-liner, starting point) ---
echo "[3/8] Installing vLLM ${RESOLVED_VLLM_VERSION} ${RESOLVED_VARIANT} from ${VLLM_INDEX} ..."
echo "  Running: uv pip install vllm --extra-index-url ${VLLM_INDEX}"
# The ROCm index is self-contained (torch, triton, amdsmi, etc.) — one extra-index covers the whole stack.
# We use --extra-index-url so the +rocm local version wins over PyPI CUDA build (pip behavior, uv also honors local version).
uv pip install --python "${PY}" vllm --extra-index-url "${VLLM_INDEX}"

# Optional but recommended on gfx1150 APU: flash-attn / aiter for attention kernel (speed, not memory)
# Warn-and-continue if unavailable — Triton fallback via FLASH_ATTENTION_TRITON_AMD_ENABLE.
echo "[3/8] Installing flash-attn + amd-aiter (optional, warn-only)..."
uv pip install --python "${PY}" --extra-index-url "${VLLM_INDEX}" "flash-attn" "amd-aiter" 2>&1 | tail -20 || echo "WARNING: flash-attn/aiter not available from ${VLLM_INDEX}, will use Triton fallback"

# --- 4. HARD VERIFICATION (no CUDA) ---
echo "[4/8] Verifying vLLM install..."
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

PKGS="$("${PY}" -m pip list --format=freeze)"
if grep -qiE '^(nvidia-|cuda-)' <<<"${PKGS}"; then
  echo "FATAL: CUDA packages found in the venv — something pulled the CUDA build of vLLM/torch." >&2
  grep -iE '^(nvidia-|cuda-)' <<<"${PKGS}" >&2 || true
  exit 1
fi

MISSING="$(ldd "${VENV_DIR}"/lib/python*/site-packages/torch/lib/*.so 2>/dev/null | grep 'not found' | sort -u || true)"
if [[ -n "${MISSING}" ]]; then
  echo "WARNING: missing shared libs (install matching apt packages):"
  echo "${MISSING}"
fi

"${PY}" - <<'EOF'
import sys, torch
print("torch        :", torch.__version__)
print("torch.hip    :", torch.version.hip)
if "rocm" not in torch.__version__ and not torch.version.hip:
    sys.exit("FATAL: torch is not a ROCm build")
if not torch.cuda.is_available():
    sys.exit("FATAL: GPU not visible to torch. Check LXC passthrough (/dev/kfd, /dev/dri), groups, and host amdgpu driver.")
print("device       :", torch.cuda.get_device_name(0))
free, total = torch.cuda.mem_get_info()
print(f"gpu memory   : {free/2**30:.1f} GiB free / {total/2**30:.1f} GiB total (iGPU = shared system RAM)")
import vllm
print("vllm         :", vllm.__version__)
try:
    import flash_attn
    print("flash-attn   :", flash_attn.__version__)
except Exception as e:
    print("flash-attn   : not installed (Triton fallback)", e)
EOF

# --- 5. OPEN-WEBUI VENV (native, same LXC, no docker) ---
echo "[5/8] Installing Open WebUI (native, port ${WEBUI_PORT})..."
rm -rf "${WEBUI_VENV_DIR}"
uv venv --seed --python "${PYTHON_VERSION}" "${WEBUI_VENV_DIR}"
WEBUI_PY="${WEBUI_VENV_DIR}/bin/python"
# open-webui pulls many deps — use PyPI (default) + same ROCm index extra if needed
uv pip install --python "${WEBUI_PY}" open-webui

"${WEBUI_PY}" -c "import open_webui; print('open-webui   :', open_webui.__version__)" 2>&1 | head -5 || "${WEBUI_PY}" -c "import importlib.metadata; print('open-webui   :', importlib.metadata.version('open-webui'))" 2>&1 | head -5 || echo "open-webui installed"

mkdir -p /opt/open-webui/data
chmod 750 /opt/open-webui/data

# --- 6. MODEL DIR ---
echo "[6/8] Checking model directory ${MODEL_DIR}..."
mkdir -p "${MODEL_DIR}/.hf-cache"
if [[ -d "${DEFAULT_MODEL_PATH}" ]]; then
  echo "Default model present: ${DEFAULT_MODEL_PATH} ($(du -sh "${DEFAULT_MODEL_PATH}" 2>/dev/null | awk '{print $1}'))"
else
  echo "WARNING: ${DEFAULT_MODEL_PATH} not found; vLLM will fail to start until it exists."
  ls -lh "${MODEL_DIR}" 2>&1 | head -30 || true
fi

# --- 7. RUNTIME CONFIG, RUNNERS, SYSTEMD UNITS ---
echo "[7/8] Writing ${ENV_FILE}, ${WEBUI_ENV_FILE}, ${RUNNER}, ${WEBUI_RUNNER}, systemd units..."
cat > "${ENV_FILE}" <<EOF
# vLLM runtime config. Edit, then: systemctl restart vllm
AI_PORT=${AI_PORT}
AI_MODEL_PATH=${DEFAULT_MODEL_PATH}
AI_SERVED_NAME=${DEFAULT_MODEL_NAME}
AI_GPU_MEM_UTIL=${GPU_MEM_UTIL}
AI_MAX_MODEL_LEN=${MAX_MODEL_LEN}
# Optional: require this bearer token on the API (recommended, host is 0.0.0.0)
AI_API_KEY=""
# Extra 'vllm serve' args, space-separated. Example: "--max-num-seqs 8 --skip-mm-profiling"
AI_EXTRA_ARGS=""
# Real vLLM variable:
VLLM_LOGGING_LEVEL=INFO
# gfx1150 override — only set if kernels fail to load with "invalid device function":
# HSA_OVERRIDE_GFX_VERSION=11.0.0
EOF
chmod 600 "${ENV_FILE}"

cat > "${WEBUI_ENV_FILE}" <<EOF
# Open WebUI runtime config. Edit, then: systemctl restart open-webui
WEBUI_PORT=${WEBUI_PORT}
# vLLM OpenAI-compatible base (WebUI talks to vLLM at 127.0.0.1:${AI_PORT})
OPENAI_API_BASE_URL=http://127.0.0.1:${AI_PORT}/v1
OPENAI_API_KEY=
# WebUI data dir (persists across updates)
DATA_DIR=/opt/open-webui/data
# Auth: BYPASS_MODEL_ACCESS_CONTROL=true lets any user see the model
BYPASS_MODEL_ACCESS_CONTROL=true
# Host bind
HOST=0.0.0.0
EOF
chmod 600 "${WEBUI_ENV_FILE}"

# vLLM runner
cat > "${RUNNER}" <<RUNNER_EOF
#!/usr/bin/env bash
# vllm-run.sh: ExecStart for vllm.service (native ROCm wheels in ${VENV_DIR})
set -euo pipefail
set -a; . ${ENV_FILE}; set +a

export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export HF_HUB_CACHE="${MODEL_DIR}/.hf-cache"
export PATH="${VENV_DIR}/bin:\${PATH}"
# Wheels bundle ROCm — do NOT set LD_LIBRARY_PATH / ROCM_PATH / HIP_PATH

ARGS=(
  "\${AI_MODEL_PATH}"
  --host 0.0.0.0
  --port "\${AI_PORT}"
  --served-model-name "\${AI_SERVED_NAME}"
  --gpu-memory-utilization "\${AI_GPU_MEM_UTIL}"
  --max-model-len "\${AI_MAX_MODEL_LEN}"
  --enforce-eager
  --trust-remote-code
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --limit-mm-per-prompt '{"image":1,"video":1}'
  --mm-processor-cache-gb 1
)
if [[ -n "\${AI_API_KEY:-}" ]]; then
  ARGS+=(--api-key "\${AI_API_KEY}")
fi
if [[ -n "\${AI_EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA <<< "\${AI_EXTRA_ARGS}"
  ARGS+=("\${EXTRA[@]}")
fi

exec ${VENV_DIR}/bin/vllm serve "\${ARGS[@]}"
RUNNER_EOF
chmod 755 "${RUNNER}"

# Open WebUI runner (native)
cat > "${WEBUI_RUNNER}" <<WEBUI_RUNNER_EOF
#!/usr/bin/env bash
# open-webui-run.sh: ExecStart for open-webui.service (native in ${WEBUI_VENV_DIR})
set -euo pipefail
set -a; . ${WEBUI_ENV_FILE}; set +a
set -a; . ${ENV_FILE}; set +a
# Ensure WebUI sees vLLM at correct port even if WEBUI_ENV_FILE uses AI_PORT
export OPENAI_API_BASE_URL="\${OPENAI_API_BASE_URL:-http://127.0.0.1:\${AI_PORT}/v1}"
export DATA_DIR="\${DATA_DIR:-/opt/open-webui/data}"
export HOST="\${HOST:-0.0.0.0}"
export PORT="\${WEBUI_PORT}"
export BYPASS_MODEL_ACCESS_CONTROL="\${BYPASS_MODEL_ACCESS_CONTROL:-true}"

export PATH="${WEBUI_VENV_DIR}/bin:\${PATH}"
mkdir -p "\${DATA_DIR}"

exec ${WEBUI_VENV_DIR}/bin/open-webui serve --host "\${HOST}" --port "\${PORT}"
WEBUI_RUNNER_EOF
chmod 755 "${WEBUI_RUNNER}"

# vLLM service
cat > "${VLLM_SERVICE}" <<UNIT
[Unit]
Description=vLLM OpenAI-compatible server (native ROCm, no docker)
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=600
StartLimitBurst=5

[Service]
Type=simple
ExecStart=${RUNNER}
Restart=on-failure
RestartSec=30
User=root

[Install]
WantedBy=multi-user.target
UNIT

# Open WebUI service (native, port 80)
cat > "${WEBUI_SERVICE}" <<UNIT
[Unit]
Description=Open WebUI (native, chat UI for vLLM)
After=network-online.target vllm.service
Wants=network-online.target
Requires=vllm.service

[Service]
Type=simple
EnvironmentFile=-${WEBUI_ENV_FILE}
EnvironmentFile=-${ENV_FILE}
ExecStart=${WEBUI_RUNNER}
Restart=on-failure
RestartSec=10
User=root

[Install]
WantedBy=multi-user.target
UNIT

# Model switch helper (vLLM)
cat > "${SWITCH_SCRIPT}" <<'EOS'
#!/usr/bin/env bash
# vllm-switch-model.sh: switch the served model (edits /etc/vllm.env, restarts service)
set -euo pipefail
ENV_FILE=/etc/vllm.env
MODEL_DIR=/srv/ai/models
PORT="$(grep '^AI_PORT=' "$ENV_FILE" | cut -d= -f2)"

echo "Current model: $(grep '^AI_MODEL_PATH=' "$ENV_FILE" | cut -d= -f2)"
echo
echo "Model dirs in ${MODEL_DIR}:"
ls -1 "${MODEL_DIR}" 2>/dev/null | grep -vE '^\.(hf-cache|cache)|^dl.*\.sh$|^switch|\.gguf$|\.bak' | sed 's/^/  /' || true
echo
read -rp "Model (dir name, full path, or HF id): " NEW_MODEL
[[ -z "${NEW_MODEL}" ]] && { echo "Aborted."; exit 1; }
if [[ "${NEW_MODEL}" != /* && "${NEW_MODEL}" != */* && -d "${MODEL_DIR}/${NEW_MODEL}" ]]; then
  NEW_MODEL="${MODEL_DIR}/${NEW_MODEL}"
fi
DEFAULT_NAME="$(basename "${NEW_MODEL}" | tr '[:upper:]' '[:lower:]')"
read -rp "Served name [${DEFAULT_NAME}]: " SERVED
SERVED="${SERVED:-${DEFAULT_NAME}}"

sed -i -E "s|^AI_MODEL_PATH=.*|AI_MODEL_PATH=${NEW_MODEL}|" "$ENV_FILE"
sed -i -E "s|^AI_SERVED_NAME=.*|AI_SERVED_NAME=${SERVED}|" "$ENV_FILE"
grep -E '^(AI_MODEL_PATH|AI_SERVED_NAME)=' "$ENV_FILE"

systemctl restart vllm
echo "Waiting for vLLM health on :${PORT} (model load can take minutes)..."
for _ in $(seq 1 90); do
  if curl -fsS -m 3 -o /dev/null "http://127.0.0.1:${PORT}/health" 2>/dev/null; then
    echo "[OK] serving ${NEW_MODEL} as ${SERVED}"
    exit 0
  fi
  sleep 2
done
echo "WARNING: not healthy yet. Check: journalctl -u vllm -f"
EOS
chmod 755 "${SWITCH_SCRIPT}"

# --- 8. ENABLE + START + VERIFY ---
echo "[8/8] Enabling and starting vllm + open-webui..."
systemctl daemon-reload
systemctl enable --now vllm
systemctl enable --now open-webui

echo "Waiting up to 5 min for vLLM health..."
HEALTHY=0
for _ in $(seq 1 60); do
  if curl -fsS -m 3 -o /dev/null "http://127.0.0.1:${AI_PORT}/health" 2>/dev/null; then
    HEALTHY=1; break
  fi
  sleep 5
done
if [[ "${HEALTHY}" == "1" ]]; then
  echo "[health] vLLM OK"
  curl -s "http://127.0.0.1:${AI_PORT}/v1/models" | head -c 800 || true
  echo
else
  echo "[health] vLLM NOT healthy yet: still loading, or a startup error. See: journalctl -u vllm -f"
fi

echo "Checking Open WebUI health (port ${WEBUI_PORT})..."
WEBUI_HEALTHY=0
for _ in $(seq 1 30); do
  if curl -fsS -m 2 -o /dev/null "http://127.0.0.1:${WEBUI_PORT}/" 2>/dev/null; then
    WEBUI_HEALTHY=1; break
  fi
  sleep 2
done
if [[ "${WEBUI_HEALTHY}" == "1" ]]; then
  echo "[health] Open WebUI OK on :${WEBUI_PORT}"
else
  echo "[health] Open WebUI not yet healthy. See: journalctl -u open-webui -f"
fi

systemctl status vllm --no-pager 2>&1 | tail -15 || true
systemctl status open-webui --no-pager 2>&1 | tail -15 || true

cat <<SUMMARY

[Bootstrap complete: vLLM ${RESOLVED_VLLM_VERSION}/${RESOLVED_VARIANT} + Open WebUI native, script v0.5.0]
  vLLM API   : http://<container-ip>:${AI_PORT}/v1   (health: /health)
  Open WebUI : http://<container-ip>:${WEBUI_PORT}/  (chat UI, BYPASS_MODEL_ACCESS_CONTROL=true)
  Model      : ${DEFAULT_MODEL_PATH} (served as ${DEFAULT_MODEL_NAME})
  Config     : ${ENV_FILE}  +  ${WEBUI_ENV_FILE}
  Switch     : ${SWITCH_SCRIPT}
  GPU        : gfx1150, ROCm via wheels ${VLLM_INDEX} (host provides amdgpu driver)
  Logs       : journalctl -u vllm -f  |  journalctl -u open-webui -f
SUMMARY
