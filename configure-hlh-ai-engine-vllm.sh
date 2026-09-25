#!/usr/bin/env bash
# configure-hlh-ai-engine-vllm.sh
# Version: 0.6.0
# Description: Native vLLM + native Open WebUI on Ubuntu 24.04 LXC (CUDA 12.8)
#              Target: NVIDIA Tesla V100 GV100GL 32GB (Volta, cc 7.0) via OCuLink eGPU.
#              No Docker — tok/s first, shared /srv/ai/models.
#
# Design rules (0.6.0 CUDA refactor):
#   1. Single source of truth for vLLM: PyPI `vllm` (the CUDA build).
#      Pinned 0.19.1 = LAST stable with torch 2.10.0+cu128 (sm_70 kernels).
#      vLLM 0.20.0+ pins torch 2.11.0+cu13 — CUDA 13.0 DROPS Volta (sm_70), and
#      driver R590+ drops Volta too, so those builds cannot run on the V100.
#   2. No Docker. Both vLLM and Open WebUI run native via uv venvs in the same LXC
#      (shared GPU, shared /srv/ai/models).
#   3. No CUDA toolkit in the LXC: the vllm/torch cu128 wheels bundle the CUDA 12.8
#      runtime (nvidia-*-cu12 pip packages). The LXC only needs driver userspace
#      matching the host driver (R580 580.65.06 = last branch for Volta):
#      libnvidia-compute-580 + nvidia-utils-580 from the CUDA ubuntu2404 repo.
#      Host kernel driver (580.65.06 on 6.14.11-9-pve) is owned by hlh-ai-engine-egpu.
#   4. Any failed check aborts loudly. No shims/fallbacks silently hiding a broken env.
#   5. Two venvs: /opt/vllm-venv (vLLM) and /opt/open-webui-venv (WebUI) — avoids dep conflicts.

set -euo pipefail

# --- CONFIGURABLE (env-overridable) ---
VLLM_VERSION="${VLLM_VERSION:-0.19.1}"         # last stable with cu128/sm_70 (see header)
NVIDIA_DRIVER_VERSION="${NVIDIA_DRIVER_VERSION:-580.65.06}"  # host driver (R580, last for Volta)
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"       # noble default; vLLM 0.19.1 needs >=3.10,<3.14
VENV_DIR="${VENV_DIR:-/opt/vllm-venv}"
WEBUI_VENV_DIR="${WEBUI_VENV_DIR:-/opt/open-webui-venv}"

AI_PORT="${AI_PORT:-8000}"
WEBUI_PORT="${WEBUI_PORT:-80}"
MODEL_DIR="${MODEL_DIR:-/srv/ai/models}"
DEFAULT_MODEL_PATH="${DEFAULT_MODEL_PATH:-${MODEL_DIR}/Qwen3.6-35B-A3B-GPTQ-Int4}"
DEFAULT_MODEL_NAME="${DEFAULT_MODEL_NAME:-qwen3.6-35b-a3b-gptq-int4}"
GPU_MEM_UTIL="${AI_GPU_MEM_UTIL:-0.85}"        # 0.85 of 32GB HBM2 when V100 is dedicated
MAX_MODEL_LEN="${AI_MAX_MODEL_LEN:-16384}"
ENABLE_ROOT_PASSWORD_SSH="${ENABLE_ROOT_PASSWORD_SSH:-1}"

VLLM_SERVICE="/etc/systemd/system/vllm.service"
WEBUI_SERVICE="/etc/systemd/system/open-webui.service"
RUNNER="/usr/local/bin/vllm-run.sh"
WEBUI_RUNNER="/usr/local/bin/open-webui-run.sh"
ENV_FILE="/etc/vllm.env"
WEBUI_ENV_FILE="/etc/open-webui.env"
SWITCH_SCRIPT="/usr/local/bin/vllm-switch-model.sh"

# --- 0. PRE-FLIGHT ---
echo "[0/8] Pre-flight checks (CUDA /dev/nvidia* passthrough)..."
if [[ ! -e /dev/nvidia0 || ! -e /dev/nvidiactl ]]; then
	echo "FATAL: /dev/nvidia0 or /dev/nvidiactl missing. Fix LXC GPU passthrough first:" >&2
	echo "  redeploy with ./deploy-hlh-ai-engine-vllm.sh (adds /dev/nvidia* bind-mounts + cgroup2 allows)" >&2
	ls -l /dev/nvidia* 2>&1 | head -10 || true
	exit 1
fi
ls -l /dev/nvidia* 2>&1 | head -10 || true

systemctl stop vllm 2>/dev/null || true
systemctl stop open-webui 2>/dev/null || true

# --- 1. BASE PACKAGES (do first so curl + network are ready) ---
echo "[1/8] Installing base packages..."
export DEBIAN_FRONTEND=noninteractive
# Wait for network briefly before apt (fresh LXC may need a few seconds after pct start)
for i in $(seq 1 12); do
	if getent hosts pypi.org >/dev/null 2>&1 || ping -c1 -W2 1.1.1.1 >/dev/null 2>&1; then
		break
	fi
	echo "  Waiting for network... ($i/12)"
	sleep 3
done
apt-get update
apt-get install -y --no-install-recommends curl ca-certificates git openssh-server \
  gnupg python3 libgomp1 2>&1 || true
# NOTE: no OpenMPI (that was a ROCm torch-wheel dep), no ROCm repo, no CUDA toolkit
# (the cu128 wheels bundle the CUDA 12.8 runtime; only driver userspace is needed below).

# --- 1. DRIVER USERSPACE (must match host R580 for the V100) ---
# Host kernel driver version is visible from the LXC via /proc/driver/nvidia/version.
echo "[1/8] Installing NVIDIA driver userspace for the V100 (host driver R580 ${NVIDIA_DRIVER_VERSION})..."
HOST_DRV="$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+' /proc/driver/nvidia/version 2>/dev/null | head -1 || true)"
if [[ -z "$HOST_DRV" ]]; then
	HOST_DRV="$NVIDIA_DRIVER_VERSION"
	echo "  /proc/driver/nvidia/version unreadable — trusting env NVIDIA_DRIVER_VERSION=${HOST_DRV}" >&2
fi
echo "  Host driver (kernel): ${HOST_DRV}"
if [[ ! "$HOST_DRV" =~ ^580\. ]]; then
	echo "FATAL: host driver ${HOST_DRV} is not R580 (580.x)." >&2
	echo "  The V100 (Volta) needs the R580 branch (580.65.06 is the last Volta driver);" >&2
	echo "  R590+ drops Volta, and R550 is below the CUDA 12.8 driver floor (570.86.15)." >&2
	echo "  Fix on the host: cd ~/git/hlh-ai-engine-egpu && ./deploy-hlh-ai-engine-egpu.sh" >&2
	exit 1
fi
DRIVER_BRANCH="${HOST_DRV%%.*}"
DRIVER_USERSPACE="${NVIDIA_DRIVER_VERSION}"

# CUDA ubuntu2404 repo (key verified BEFORE list written; repair stale list without key)
mkdir -p /usr/share/keyrings
if [ -f /etc/apt/sources.list.d/cuda-ubuntu2404.list ] && [ ! -s /usr/share/keyrings/cuda-ubuntu2404.gpg ]; then
	echo "  Removing stale cuda-ubuntu2404.list (missing keyring)..."
	rm -f /etc/apt/sources.list.d/cuda-ubuntu2404.list
fi
if [ ! -s /usr/share/keyrings/cuda-ubuntu2404.gpg ]; then
	echo "  Adding CUDA ubuntu2404 repo (driver userspace ${DRIVER_BRANCH} branch)..."
	_tmp_pub="$(mktemp)"
	if ! curl -fsSL --retry 3 --retry-delay 5 \
		"https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub" -o "$_tmp_pub"; then
		echo "ERROR: Failed to fetch CUDA ubuntu2404 key" >&2; rm -f "$_tmp_pub"; exit 1
	fi
	[ -s "$_tmp_pub" ] || { echo "ERROR: downloaded CUDA key is empty" >&2; rm -f "$_tmp_pub"; exit 1; }
	gpg --batch --yes --dearmor -o /usr/share/keyrings/cuda-ubuntu2404.gpg "$_tmp_pub" || { echo "ERROR: gpg dearmor failed" >&2; rm -f "$_tmp_pub"; exit 1; }
	rm -f "$_tmp_pub"
	[[ -s /usr/share/keyrings/cuda-ubuntu2404.gpg ]] || { echo "ERROR: CUDA keyring empty after dearmor" >&2; exit 1; }
	echo "deb [signed-by=/usr/share/keyrings/cuda-ubuntu2404.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64 /" > /etc/apt/sources.list.d/cuda-ubuntu2404.list
fi
apt-get update || { echo "ERROR: apt-get update failed after adding CUDA repo" >&2; exit 1; }

echo "  Installing libnvidia-compute-${DRIVER_BRANCH} + nvidia-utils-${DRIVER_BRANCH} ${DRIVER_USERSPACE} (provides libcuda.so.1, libnvidia-ml.so, nvidia-smi)..."
apt-get install -y --allow-downgrades \
	"libnvidia-compute-${DRIVER_BRANCH}=${DRIVER_USERSPACE}-0ubuntu1" \
	"nvidia-utils-${DRIVER_BRANCH}=${DRIVER_USERSPACE}-0ubuntu1" 2>&1 | tail -n 20 || \
apt-get install -y --allow-downgrades \
	"libnvidia-compute-${DRIVER_BRANCH}" "nvidia-utils-${DRIVER_BRANCH}" 2>&1 | tail -n 20
apt-mark hold "libnvidia-compute-${DRIVER_BRANCH}" "nvidia-utils-${DRIVER_BRANCH}" 2>&1 | head -n 5 || true

# Hard gates: nvidia-smi sees the V100, libcuda.so.1 resolves
set +o pipefail
nvidia-smi -L 2>&1 | head -5 || true
set -o pipefail
GPU_COUNT="$(nvidia-smi -L 2>&1 | grep -c 'GPU [0-9]:' || true)"
if [[ "$GPU_COUNT" -ne 1 ]]; then
	echo "FATAL: expected exactly 1 V100 GPU inside the LXC, found ${GPU_COUNT}." >&2
	nvidia-smi -L 2>&1 | head -10 || true
	exit 1
fi
# NOTE: this GV100GL board reports its VBIOS product name "Tesla PG500-216" in
# nvidia-smi (NOT "Tesla V100") — the torch capability check below is the
# authoritative sm_70 gate.
echo "  GPU: $(nvidia-smi -L 2>/dev/null | head -1 || true)"
ldconfig
if ! ldconfig -p 2>/dev/null | grep -q 'libcuda.so.1'; then
	echo "FATAL: libcuda.so.1 not resolvable after userspace install (libnvidia-compute-${DRIVER_BRANCH})." >&2
	exit 1
fi
echo "  Driver userspace OK: $(nvidia-smi -L | head -1)"

# VRAM preflight: the V100's 32GB HBM2 is shared with LXC 111 (llama.cpp).
# vLLM needs ~AI_GPU_MEM_UTIL of total VRAM free at startup.
MEM_ROW="$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
MEM_USED="$(printf '%s\n' "${MEM_ROW}" | awk -F', *' '{print $1}')"
MEM_TOTAL="$(printf '%s\n' "${MEM_ROW}" | awk -F', *' '{print $2}')"
if [[ -n "${MEM_USED}" && -n "${MEM_TOTAL}" && "${MEM_TOTAL}" -gt 0 ]]; then
	MEM_FREE=$((MEM_TOTAL - MEM_USED))
	NEED_FREE="$(awk -v t="${MEM_TOTAL}" -v u="${GPU_MEM_UTIL}" 'BEGIN{printf "%d", t*u}')"
	echo "  VRAM: ${MEM_USED} MiB used / ${MEM_TOTAL} MiB (${MEM_FREE} MiB free); vLLM needs ~${NEED_FREE} MiB free (AI_GPU_MEM_UTIL=${GPU_MEM_UTIL})"
	if [[ "${MEM_FREE}" -lt "${NEED_FREE}" && "${SKIP_VRAM_PREFLIGHT:-0}" != "1" ]]; then
		echo "FATAL: not enough free VRAM on the shared V100 (${MEM_FREE} MiB free < ~${NEED_FREE} MiB needed)." >&2
		echo "  Another engine is using ${MEM_USED} MiB — on this host that is LXC 111 (hlh-ai-engine-egpu, llama.cpp)." >&2
		echo "  Stop it first:  pct exec 111 -- systemctl stop ai-engine" >&2
		echo "  (override: SKIP_VRAM_PREFLIGHT=1, or lower AI_GPU_MEM_UTIL for co-tenancy)." >&2
		exit 1
	fi
fi

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

# --- 3. INSTALL vLLM (PyPI CUDA build) ---
echo "[3/8] Installing vLLM ${VLLM_VERSION} (PyPI CUDA build -> torch 2.10.0+cu128) ..."
# The PyPI `vllm` wheel IS the CUDA build. torch 2.10.0 (pinned by vLLM 0.19.1) pulls
# the CUDA 12.8 runtime via nvidia-*-cu12 pip packages — no CUDA toolkit apt packages needed.
uv pip install --python "${PY}" "vllm==${VLLM_VERSION}"
# /metrics endpoint (prometheus_client is a hard dep of vllm 0.19.x; keep belt-and-suspenders)
uv pip install --python "${PY}" prometheus_client 2>&1 | tail -5 || true

# --- 4. HARD VERIFICATION (CUDA 12.8 on the V100) ---
echo "[4/8] Verifying vLLM install (expect torch 2.10.0 cu12.8 + Tesla V100 32GB)..."
PKGS="$("${PY}" -m pip list --format=freeze)"
if grep -qiE '\+rocm' <<<"${PKGS}"; then
	echo "FATAL: ROCm packages found in the venv — something pulled the ROCm build." >&2
	grep -iE '\+rocm' <<<"${PKGS}" >&2 || true
	exit 1
fi
if grep -qiE '^nvidia-.*cu13' <<<"${PKGS}"; then
	echo "FATAL: CUDA 13 runtime packages found — the cu13 build has no sm_70 (Volta) kernels." >&2
	grep -iE '^nvidia-.*cu13' <<<"${PKGS}" >&2 || true
	exit 1
fi

"${PY}" - <<'EOF'
import sys, torch
print("torch        :", torch.__version__)
if torch.version.cuda != "12.8":
    sys.exit(f"FATAL: expected torch cu12.8, got cu{torch.version.cuda} (V100 Volta needs CUDA 12.x)")
if not torch.cuda.is_available():
    sys.exit("FATAL: GPU not visible to torch. Check LXC passthrough (/dev/nvidia0, /dev/nvidiactl, /dev/nvidia-uvm).")
name = torch.cuda.get_device_name(0)
print("device       :", name)
cc = torch.cuda.get_device_capability(0)
print("capability   :", cc)
if cc != (7, 0):
    sys.exit(f"FATAL: expected Volta sm_70 (7,0), got {cc} on {name}")
free, total = torch.cuda.mem_get_info()
print(f"gpu memory   : {free/2**30:.1f} GiB free / {total/2**30:.1f} GiB total (V100 = 32GB HBM2)")
# Smoke: a real kernel launch on sm_70 (catches 'no kernel image available' early)
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = a @ a
torch.cuda.synchronize()
print("cuda matmul  : OK (fp16, sm_70)")
import vllm
print("vllm         :", vllm.__version__)
EOF

# --- 5. OPEN-WEBUI VENV (native, same LXC, no docker) ---
echo "[5/8] Installing Open WebUI (native, port ${WEBUI_PORT})..."
rm -rf "${WEBUI_VENV_DIR}"
uv venv --seed --python "${PYTHON_VERSION}" "${WEBUI_VENV_DIR}"
WEBUI_PY="${WEBUI_VENV_DIR}/bin/python"
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
# Prometheus metrics always exposed at http://${AI_PORT}/metrics (prometheus_client, no flag needed)
# Optional: require this bearer token on the API (recommended, host is 0.0.0.0)
AI_API_KEY=""
# Extra 'vllm serve' args, space-separated. Example: "--max-num-seqs 8"
AI_EXTRA_ARGS=""
# Real vLLM variable:
VLLM_LOGGING_LEVEL=INFO
# --- V100 (Volta sm_70) notes ---
# Compute dtype is fp16 (vLLM avoids bf16 on CC<8.0). --enforce-eager is on by default
# in /usr/local/bin/vllm-run.sh (safe on Volta); remove it there to try CUDA graphs.
# The V100's 32GB HBM2 is SHARED with LXC 111 (hlh-ai-engine-egpu, llama.cpp): if both
# engines run, lower AI_GPU_MEM_UTIL (e.g. 0.45) or stop the sibling first.
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
# vllm-run.sh: ExecStart for vllm.service (native CUDA 12.8 wheels in ${VENV_DIR})
set -euo pipefail
set -a; . ${ENV_FILE}; set +a

export HF_HUB_CACHE="${MODEL_DIR}/.hf-cache"
export PATH="${VENV_DIR}/bin:\${PATH}"

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
Description=vLLM OpenAI-compatible server (native CUDA 12.8 on Tesla V100, no docker)
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
Wants=vllm.service

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

# Clean up ROCm-era leftovers from the 0.5.x stack (idempotent)
rm -f /etc/apt/sources.list.d/rocm.list /etc/apt/preferences.d/rocm-pin /etc/ld.so.conf.d/rocm.conf 2>/dev/null || true
rm -f /etc/apt/keyrings/amdrocm.gpg 2>/dev/null || true
ldconfig 2>/dev/null || true

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

[Bootstrap complete: vLLM ${VLLM_VERSION} CUDA 12.8 (V100 sm_70) + Open WebUI native, script v0.6.0]
  vLLM API   : http://<container-ip>:${AI_PORT}/v1   (health: /health)
  Open WebUI : http://<container-ip>:${WEBUI_PORT}/  (chat UI, BYPASS_MODEL_ACCESS_CONTROL=true)
  Model      : ${DEFAULT_MODEL_PATH} (served as ${DEFAULT_MODEL_NAME})
  Config     : ${ENV_FILE}  +  ${WEBUI_ENV_FILE}
  Switch     : ${SWITCH_SCRIPT}
  GPU        : Tesla V100 32GB HBM2 (Volta sm_70), driver userspace ${DRIVER_BRANCH} ${DRIVER_USERSPACE}
  Logs       : journalctl -u vllm -f  |  journalctl -u open-webui -f  |  nvidia-smi
SUMMARY
