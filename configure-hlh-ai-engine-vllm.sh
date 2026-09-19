#!/usr/bin/env bash
# configure-hlh-ai-engine-vllm.sh
# Version: 0.4.0
# Description: Native vLLM (no docker) in an Ubuntu 24.04 LXC using AMD's prebuilt
#              ROCm 10.0.0 wheels. Target: Radeon 890M (gfx1150 / Strix Point).
#
# Design rules (learned the hard way in 0.3.x):
#   1. Only AMD's ROCm wheels. NEVER `pip install vllm` from PyPI (that is the CUDA build).
#   2. No source build, no site-packages patching, no shims, no fallbacks.
#   3. Any failed check aborts the run. A broken env must fail loudly, not "sort of work".
#   4. ROCm userspace comes from the pip wheels (torch pulls in the ROCm core libs).
#      Do NOT also install ROCm via apt or put /opt/rocm on LD_LIBRARY_PATH.
#   5. The LXC only needs: /dev/kfd + /dev/dri passthrough, and a recent amdgpu driver
#      on the Proxmox host (the container shares the host kernel).
#
# Wheel versions come from AMD's docs (rocm.docs.amd.com -> AI Ecosystem -> vLLM).
# Re-check them when upgrading; AMD publishes new builds regularly.

set -euo pipefail

# --- CONFIGURABLE (env-overridable) ---
GFX="${GFX:-gfx1150}"
PYTHON_VERSION="${PYTHON_VERSION:-3.14}"
VENV_DIR="${VENV_DIR:-/opt/vllm-venv}"

TORCH_SPEC="${TORCH_SPEC:-2.12.0+rocm10.0.0}"          # gfx1151 uses 2.13.0 / torchvision 0.28.0
TORCHVISION_SPEC="${TORCHVISION_SPEC:-0.27.0+rocm10.0.0}"
TORCHAUDIO_SPEC="${TORCHAUDIO_SPEC:-2.11.0+rocm10.0.0}"
TORCH_INDEX="https://stable.repo.amd.com/rocm/whl-next/"
VLLM_EXTRA_INDEX="https://rocm.frameworks.amd.com/whl-multi-arch/vllm/"
VLLM_WHEEL="${VLLM_WHEEL:-https://rocm.frameworks.amd.com/whl-multi-arch/vllm/vllm/vllm-0.27.1.dev5%2Brocm10.0.0.gf46a9dfe2.d20260826-cp314-cp314-linux_x86_64.whl}"

AI_PORT="${AI_PORT:-8000}"
MODEL_DIR="${MODEL_DIR:-/srv/ai/models}"
DEFAULT_MODEL_PATH="${DEFAULT_MODEL_PATH:-${MODEL_DIR}/Qwen3.5-9B}"
DEFAULT_MODEL_NAME="${DEFAULT_MODEL_NAME:-qwen3.5-9b}"
GPU_MEM_UTIL="${AI_GPU_MEM_UTIL:-0.40}"
MAX_MODEL_LEN="${AI_MAX_MODEL_LEN:-4096}"
ENABLE_ROOT_PASSWORD_SSH="${ENABLE_ROOT_PASSWORD_SSH:-1}"   # set 0 to skip the root/password SSH config

VLLM_SERVICE="/etc/systemd/system/vllm.service"
RUNNER="/usr/local/bin/vllm-run.sh"
ENV_FILE="/etc/vllm.env"
SWITCH_SCRIPT="/usr/local/bin/vllm-switch-model.sh"
AMD_SMI_PATH="${VENV_DIR}/lib/python${PYTHON_VERSION}/site-packages/_rocm_sdk_core/share/amd_smi"

# --- 0. PRE-FLIGHT: fail early, before a long install ---
echo "[0/7] Pre-flight checks..."
if [[ ! -e /dev/kfd ]]; then
  echo "FATAL: /dev/kfd missing. Fix LXC GPU passthrough (/dev/kfd + /dev/dri) first." >&2
  exit 1
fi
ls -l /dev/kfd /dev/dri/ 2>&1 | head -10 || true

systemctl stop vllm 2>/dev/null || true
systemctl disable vllm 2>/dev/null || true

# --- 1. BASE PACKAGES ---
echo "[1/7] Installing base packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends curl ca-certificates git openssh-server \
  libgomp1 libnuma1 libatomic1 libdrm2

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

# --- 2. uv + PYTHON 3.14 VENV (AMD's wheels are cp314; Ubuntu 24.04 ships 3.12) ---
echo "[2/7] Installing uv and creating Python ${PYTHON_VERSION} venv at ${VENV_DIR}..."
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
fi
export PATH="/usr/local/bin:${PATH}"

rm -rf "${VENV_DIR}"                      # always rebuild clean; leftover CUDA pieces cause chaos
uv venv --seed --python "${PYTHON_VERSION}" "${VENV_DIR}"
PY="${VENV_DIR}/bin/python"

# --- 3. INSTALL: torch (ROCm) -> flash-attn/aiter -> vLLM (ROCm) ---
echo "[3/7] Installing PyTorch ${TORCH_SPEC} for ${GFX} from AMD's index..."
"${PY}" -m pip install --index-url "${TORCH_INDEX}" \
  "torch[device-${GFX}]==${TORCH_SPEC}" \
  "torchvision[device-${GFX}]==${TORCHVISION_SPEC}" \
  "torchaudio==${TORCHAUDIO_SPEC}"

echo "[3/7] Installing flash-attn + AITER..."
"${PY}" -m pip install --extra-index-url "${VLLM_EXTRA_INDEX}" \
  "flash-attn==2.8.3" "amd-aiter==0.1.20.post1"

echo "[3/7] Installing AMD's ROCm vLLM wheel (uv resolves deps predictably)..."
uv pip install --python "${PY}" "${VLLM_WHEEL}"

echo "[3/7] tensorizer workaround from AMD's known-issues list..."
"${PY}" -m pip install --upgrade "tensorizer==2.12.1"

# --- 4. HARD VERIFICATION (no fallbacks: any failure aborts) ---
echo "[4/7] Verifying install..."
export PYTHONPATH="${AMD_SMI_PATH}"
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
[[ -d "${AMD_SMI_PATH}" ]] || echo "WARNING: ${AMD_SMI_PATH} not found; check the docs for the current amd_smi path."

PKGS="$("${PY}" -m pip list --format=freeze)"
if grep -qiE '^(nvidia-|cuda-)' <<<"${PKGS}"; then
  echo "FATAL: CUDA packages found in the venv. Something pulled the CUDA build of vLLM/torch." >&2
  grep -iE '^(nvidia-|cuda-)' <<<"${PKGS}" >&2 || true
  exit 1
fi

# List ALL missing shared libs up front (warning only; the python check below is the real gate)
MISSING="$(ldd "${VENV_DIR}"/lib/python${PYTHON_VERSION}/site-packages/torch/lib/*.so 2>/dev/null | grep 'not found' | sort -u || true)"
if [[ -n "${MISSING}" ]]; then
  echo "WARNING: missing shared libraries (install matching apt packages):"
  echo "${MISSING}"
fi

"${PY}" - <<'EOF'
import sys, torch
print("torch        :", torch.__version__)
print("torch.hip    :", torch.version.hip)
if "rocm" not in torch.__version__ or not torch.version.hip:
    sys.exit("FATAL: torch is not a ROCm build")
if not torch.cuda.is_available():
    sys.exit("FATAL: GPU not visible to torch. Check LXC passthrough (/dev/kfd, /dev/dri), groups, and host amdgpu driver.")
print("device       :", torch.cuda.get_device_name(0))
free, total = torch.cuda.mem_get_info()
print(f"gpu memory   : {free/2**30:.1f} GiB free / {total/2**30:.1f} GiB total (iGPU = shared system RAM)")
import vllm
print("vllm         :", vllm.__version__)
import flash_attn
print("flash-attn   :", flash_attn.__version__)
EOF

# --- 5. MODEL DIR ---
echo "[5/7] Checking model directory ${MODEL_DIR}..."
mkdir -p "${MODEL_DIR}/.hf-cache"
if [[ -d "${DEFAULT_MODEL_PATH}" ]]; then
  echo "Default model present: ${DEFAULT_MODEL_PATH} ($(du -sh "${DEFAULT_MODEL_PATH}" 2>/dev/null | awk '{print $1}'))"
else
  echo "WARNING: ${DEFAULT_MODEL_PATH} not found; vLLM will fail to start until it exists."
  ls -lh "${MODEL_DIR}" 2>&1 | head -30 || true
fi

# --- 6. RUNTIME CONFIG, RUNNER, SYSTEMD UNIT ---
echo "[6/7] Writing ${ENV_FILE}, ${RUNNER}, ${VLLM_SERVICE}..."
# NOTE: variables use the AI_ prefix on purpose. vLLM warns on unknown VLLM_* variables.
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
# Do NOT set HSA_OVERRIDE_GFX_VERSION: gfx1150 is supported by the ROCm 10 wheels.
# Only as a last resort if kernels fail to load: HSA_OVERRIDE_GFX_VERSION=11.0.0
EOF
chmod 600 "${ENV_FILE}"

cat > "${RUNNER}" <<RUNNER_EOF
#!/usr/bin/env bash
# vllm-run.sh: ExecStart for vllm.service (native vLLM, ROCm wheels in ${VENV_DIR})
set -euo pipefail
set -a; . ${ENV_FILE}; set +a

export PYTHONPATH="${AMD_SMI_PATH}"
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export HF_HUB_CACHE="${MODEL_DIR}/.hf-cache"
export PATH="${VENV_DIR}/bin:\${PATH}"
# Deliberately NOT setting LD_LIBRARY_PATH / ROCM_PATH / HIP_PATH: the wheels bundle ROCm.

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

# 'vllm serve' honors the positional model; the deprecated api_server module did not.
exec ${VENV_DIR}/bin/vllm serve "\${ARGS[@]}"
RUNNER_EOF
chmod 755 "${RUNNER}"

cat > "${VLLM_SERVICE}" <<UNIT
[Unit]
Description=vLLM OpenAI-compatible server (native ROCm, no docker)
After=network-online.target
Wants=network-online.target
# Stop the endless restart loop: max 5 failures per 10 minutes, then give up.
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

# --- model switch helper ---
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

# --- 7. ENABLE + START + VERIFY ---
echo "[7/7] Enabling and starting vllm..."
systemctl daemon-reload
systemctl enable --now vllm

echo "Waiting up to 5 min for health..."
HEALTHY=0
for _ in $(seq 1 60); do
  if curl -fsS -m 3 -o /dev/null "http://127.0.0.1:${AI_PORT}/health" 2>/dev/null; then
    HEALTHY=1; break
  fi
  sleep 5
done
if [[ "${HEALTHY}" == "1" ]]; then
  echo "[health] OK"
  curl -s "http://127.0.0.1:${AI_PORT}/v1/models" | head -c 800 || true
  echo
else
  echo "[health] NOT healthy yet: still loading, or a startup error. See: journalctl -u vllm -f"
fi
systemctl status vllm --no-pager 2>&1 | tail -15 || true

cat <<SUMMARY

[Bootstrap complete: vLLM native ROCm, script v0.4.0]
  API        : http://<container-ip>:${AI_PORT}/v1   (health: /health)
  Model      : ${DEFAULT_MODEL_PATH} (served as ${DEFAULT_MODEL_NAME})
  Config     : ${ENV_FILE}  (edit, then: systemctl restart vllm)
  Switch     : ${SWITCH_SCRIPT}
  GPU        : ${GFX}, ROCm 10.0.0 via pip wheels; host provides amdgpu driver
  Logs       : journalctl -u vllm -f
SUMMARY
