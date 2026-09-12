#!/usr/bin/env bash
# configure-ai-engine-inside-lxc.sh (vLLM variant)
# Version: 0.1.3
# Description: Bootstrap vLLM + Open WebUI on Ubuntu 24.04 LXC with ROCm passthrough (gfx1150)
# Target GPU: AMD Radeon 890M (gfx1150/Strix Halo) on Proxmox 9.x privileged LXC — mirrors hlh-ai-engine 113/192.168.1.13
# Requirements: Run as root inside privileged LXC with GPU passthrough (/dev/dri/card1, renderD129, /dev/kfd) and /srv/ai/models bind mount
# Changelog:
#   0.1.3 - Fix Open WebUI privileged LXC AppArmor (add --security-opt apparmor=unconfined), move WEBUI_PORT 8080 -> 80, default model /srv/ai/models/Qwen3.5-9B-safetensors (qwen3.5-9b, 0.70/4096+mm), mirror live 113 flags
#   0.1.2 - Fix vLLM bootstrap: HF Qwen2.5-7B (not GGUF qwen35moe which 0.6.6 cannot load), GFX 11.0.0 override (11.5.0 gives HIP invalid device), ld.so.conf for libamd_smi, amdsmi 27.0.0
#   0.1.1 - Default model now shared GGUF /srv/ai/models/Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf (same as hlh-ai-engine 112, 98304 ctx) — added hf-cache check, VLLM_LOGGING_LEVEL=DEBUG, pre-check rocm-smi (reverted: GGUF not supported by vLLM 0.6.6)
#   0.1.0 - Initial vLLM variant forked from hlh-ai-engine v0.9.4
#           ROCm 10.0.0 default never pinned (ROCM_VERSION env, 7.14.1 rollback supported)
#           Replaces llama.cpp HIP+Vulkan dual build with vLLM ROCm (pip) + Open WebUI (docker)
#           Model storage /srv/ai/models shared with siblings (GGUF for llama.cpp, HF safetensors for vLLM)

set -euo pipefail

# --- CONFIGURABLE ---
MODEL_DIR="/srv/ai/models"
# Default model is now local safetensors at /srv/ai/models/Qwen3.5-9B-safetensors (qwen3.5-9b)
# Previous HF defaults Qwen/Qwen2.5-7B-Instruct (qwen2.5-7b) kept as fallback via vllm-switch-model.sh.
# HF cache is at /srv/ai/models/.hf-cache (shared). GGUF Qwen3.6-35B qwen35moe remains for llama.cpp sibling (vLLM 0.29.0 cannot load GGUF qwen35moe).
DEFAULT_MODEL_HF="/srv/ai/models/Qwen3.5-9B-safetensors"
DEFAULT_MODEL_NAME="qwen3.5-9b"
VENV_DIR="/opt/vllm-venv"
ROCM_PATH="/opt/rocm"
ROCM_VERSION="${ROCM_VERSION:-10.0.0}"
GFX_VERSION="11.0.0"   # gfx1150 via HSA_OVERRIDE 11.0.0 for torch ROCm 6.2/6.4 (11.5.0 gave HIP invalid device with 2.5.1+rocm6.2 on 0.6.6)
VLLM_PORT="8000"
WEBUI_PORT="80"
VLLM_SERVICE="/etc/systemd/system/vllm.service"
WEBUI_SERVICE="/etc/systemd/system/open-webui.service"
SWITCH_SCRIPT="/usr/local/bin/vllm-switch-model.sh"

# --- 1. BASE DEPENDENCIES + ROCm ---
echo "[1/6] Installing base dependencies (ROCm ${ROCM_VERSION}, gfx1150, vLLM + Open WebUI)..."
apt-get update
apt-get install -y --no-install-recommends \
  build-essential git curl wget ca-certificates gnupg \
  python3 python3-venv python3-pip python3-dev \
  libopenblas-dev libssl-dev pkg-config \
  openssh-server docker.io \
  libnuma1 2>&1 || true

# Host driver must match LXC user-space major — deploy script already upgraded host if user answered 'y'.
# Add ROCm repo (mirrors hlh-ai-engine logic: stable for 10.x, legacy for 7.x)
echo "[1/6] Adding ROCm ${ROCM_VERSION} repository..."
mkdir -p /etc/apt/keyrings
ROCM_MAJOR="$(echo "${ROCM_VERSION}" | cut -d. -f1)"
if [ "${ROCM_MAJOR}" -ge 10 ] 2>/dev/null; then
  wget -qO - https://stable.repo.amd.com/rocm/gpg/packages.gpg | gpg --dearmor | tee /etc/apt/keyrings/amdrocm.gpg > /dev/null
  tee /etc/apt/sources.list.d/rocm.list << EOF
deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] https://stable.repo.amd.com/rocm/core/packages/ubuntu2404 stable main
EOF
  tee /etc/apt/preferences.d/rocm-pin << 'PIN'
Package: *
Pin: origin stable.repo.amd.com
Pin-Priority: 1001
PIN
else
  wget -qO - https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg | gpg --dearmor | tee /etc/apt/keyrings/amdrocm.gpg > /dev/null
  tee /etc/apt/sources.list.d/rocm.list << EOF
deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] https://repo.amd.com/rocm/packages-multi-arch/ubuntu2404 stable main
EOF
  tee /etc/apt/preferences.d/rocm-pin << 'PIN'
Package: *
Pin: origin repo.radeon.com
Pin-Priority: 1001
PIN
fi
echo 'APT::Key::GPGCommand "/usr/bin/gpg";' > /etc/apt/apt.conf.d/99gpg-override || true
apt-get remove -y rocminfo 2>/dev/null || true
apt-get update
ROCM_MM="$(echo "${ROCM_VERSION}" | cut -d. -f1,2)"
echo "[1/6] Installing ROCm ${ROCM_VERSION} packages: amdrocm${ROCM_MM}-gfx1150 ..."
if ! apt-get install -y --no-install-recommends "amdrocm${ROCM_MM}-gfx1150" "amdrocm-core-dev${ROCM_MM}-gfx1150" 2>&1; then
  echo "WARNING: per-GPU package not found, trying generic amdrocm${ROCM_MM} ..."
  apt-get install -y --no-install-recommends "amdrocm${ROCM_MM}" "amdrocm-core-dev${ROCM_MM}" 2>&1 || {
    echo "Trying generic amdrocm ..."
    apt-get install -y --no-install-recommends amdrocm 2>&1 || true
  }
fi

# Verify HIP
if [ ! -f /opt/rocm/lib/cmake/hip-lang/hip-lang-config.cmake ] && [ ! -f /opt/rocm/lib64/cmake/hip-lang/hip-lang-config.cmake ]; then
  echo "WARNING: HIP CMake package not found after ROCm install (may still work for vLLM pip wheels)"
fi

# Fix ld.so for ROCm 10 amdsmi (libamd_smi.so not in cache without rocm.conf)
if [ ! -f /etc/ld.so.conf.d/rocm.conf ]; then
  echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf
  echo "/opt/rocm/lib64" >> /etc/ld.so.conf.d/rocm.conf 2>&1 || true
  ldconfig 2>&1 | head -5 || true
fi
ldconfig -p 2>&1 | grep -q libamd_smi.so || ldconfig 2>&1 | head -5 || true

# Add root to render/video (needed for /dev/kfd rw — crw-rw---- root render)
usermod -aG render root 2>&1 || true
usermod -aG video root 2>&1 || true
# Ensure render group membership takes effect for systemd (Group=render in service)

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

# ROCm env
tee /etc/profile.d/rocm.env << EOF
export PATH=\$PATH:${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin
export LD_LIBRARY_PATH=${ROCM_PATH}/lib:\${LD_LIBRARY_PATH:-}
export ROCM_PATH=${ROCM_PATH}
export HIP_PATH=${ROCM_PATH}
export HSA_OVERRIDE_GFX_VERSION=${GFX_VERSION}
EOF
set +u; source /etc/profile.d/rocm.env; set -u

# Docker for Open WebUI
systemctl enable docker 2>&1 || true
systemctl start docker 2>&1 || true
docker --version 2>&1 | head -1 || echo "WARNING: docker not ready"

# --- 2. Python venv + vLLM ---
echo "[2/6] Creating Python venv at ${VENV_DIR} and installing vLLM (ROCm)..."
python3 -m venv "${VENV_DIR}" 2>&1 || true
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip wheel setuptools 2>&1 | tail -20

# vLLM ROCm install: use stable ROCm wheels. gfx1150 (APU) is community-tier; try official pip first, fallback to ROCm extra-index.
# Host is trixie debian13 with ROCm 10.0 stable; LXC is ubuntu2404. Need torch+ROCm matching host driver (10.0) or vLLM falls back to CPU and fails "Failed to infer device type".
echo "[2/6] Installing vLLM (this can take 10-20 minutes)..."
# quick device check before pip — correct card is card0/renderD128 for 890M (not card1)
echo "[2/6] Pre-check: rocm-smi + hip + /dev/kfd"
rocm-smi 2>&1 | head -30 || echo "WARNING: rocm-smi no GPUs (host/LXC ROCm mismatch?)"
rocminfo 2>&1 | head -30 || true
ls -l /dev/kfd /dev/dri/card0 /dev/dri/renderD128 2>&1 | head -10 || true
# Ensure kfd is accessible (needs render group)
sg render -c "rocm-smi" 2>&1 | head -20 || true
# Install torch ROCm first so vLLM picks ROCm torch, not CUDA/CPU. Use rocm6.4 for newer vLLM (0.29.0 needs torch 2.8+), rocm6.2 for 0.6.6 needs 2.5
pip install --index-url https://download.pytorch.org/whl/rocm6.4 "torch==2.8.0+rocm6.4" "torchvision==0.23.0+rocm6.4" 2>&1 | tail -50 || \
  pip install --extra-index-url https://download.pytorch.org/whl/rocm6.2 torch torchvision --upgrade 2>&1 | tail -30 || true
# Fix amdsmi for ROCm 10: pip amdsmi 7.0.2 is for ROCm 7, need 27.0.0 from /opt/rocm/share/amd_smi
if pip show amdsmi 2>&1 | grep -q "7.0.2"; then
  echo "[2/6] Replacing pip amdsmi 7.0.2 with ROCm 10 amdsmi from /opt/rocm/share/amd_smi ..."
  pip uninstall -y amdsmi 2>&1 | tail -10 || true
  pip install /opt/rocm/share/amd_smi 2>&1 | tail -30 || true
fi
# Ensure vLLM version compatible with torch+transformers. vLLM 0.29.0 needs transformers>=5.10 and torch 2.8+rocm6.4
echo "[2/6] Installing vLLM 0.29.0 (ROCm) — GGUF qwen35moe not supported yet, HF Qwen2.5-7B is default"
pip install "vllm==0.29.0" 2>&1 | tail -50 || pip install "vllm[rocm]" 2>&1 | tail -50 || true
if ! "${VENV_DIR}/bin/python" -c "import vllm; print(vllm.__version__)" 2>&1 | head -5; then
  echo "Retrying generic vllm install..."
  pip install vllm 2>&1 | tail -50 || true
fi
"${VENV_DIR}/bin/python" -c "import vllm; print('vLLM', vllm.__version__)" 2>&1 | head -5 || echo "WARNING: vLLM import failed — may need ROCm-specific wheel"
# Pin transformers to 5.17 for vLLM 0.29.0 (4.45 too old, 5.17 works for Qwen2.5-7B)
pip install "transformers==5.17.0" 2>&1 | tail -20 || true
# Debug device inference right after install
VLLM_LOGGING_LEVEL=DEBUG "${VENV_DIR}/bin/python" -c "import os; os.environ['HSA_OVERRIDE_GFX_VERSION']='11.0.0'; from vllm.platforms import rocm; print('rocm platform check done')" 2>&1 | head -30 || true
deactivate

# --- 3. MODEL DIR ---
echo "[3/6] Setting up model directory ${MODEL_DIR}..."
mkdir -p "${MODEL_DIR}"
# Shared storage: /srv/ai/models holds both GGUF (llama.cpp) and safetensors (vLLM). Default is now local safetensors /srv/ai/models/Qwen3.5-9B-safetensors.
mkdir -p /root/.cache/huggingface 2>&1 || true
mkdir -p "${MODEL_DIR}/.hf-cache" 2>&1 || true
if [ -e "${DEFAULT_MODEL_HF}" ]; then
  echo "Default model present: ${DEFAULT_MODEL_HF} ($(du -sh "${DEFAULT_MODEL_HF}" 2>&1 | awk '{print $1}'))"
  ls -lh "${DEFAULT_MODEL_HF}" 2>&1 | head -20 || true
else
  echo "WARNING: Default model ${DEFAULT_MODEL_HF} not found on ${MODEL_DIR} — vLLM will fail to start until model is present."
  echo "Available models in ${MODEL_DIR}:"
  ls -lh "${MODEL_DIR}" 2>&1 | head -30 || true
fi
# Create a helper for HF download (mirrors dl.sh but for HF)
cat > "${MODEL_DIR}/dl-hf.sh" << 'EOS'
#!/usr/bin/env bash
# dl-hf.sh - HuggingFace HF Hub downloader via huggingface-cli or curl
# Usage: dl-hf.sh <model-id>  e.g. Qwen/Qwen2.5-Coder-32B-Instruct
set -euo pipefail
MODEL_ID="${1:-}"
if [ -z "$MODEL_ID" ]; then echo "Usage: $0 <hf-model-id>"; exit 1; fi
echo "Downloading HF model: $MODEL_ID (uses huggingface cache)"
if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$MODEL_ID" --repo-type model || echo "huggingface-cli download failed, vLLM will download on first serve"
else
  echo "huggingface-cli not found, installing huggingface_hub..."
  pip install -q huggingface_hub 2>&1 | tail -5 || true
  python3 -m huggingface_hub.commands.huggingface_cli download "$MODEL_ID" --repo-type model || true
fi
EOS
chmod +x "${MODEL_DIR}/dl-hf.sh"

# --- 4. vLLM systemd service ---
echo "[4/6] Creating vLLM systemd service (${VLLM_SERVICE}) on :${VLLM_PORT}..."
cat > "${VLLM_SERVICE}" << UNIT
[Unit]
Description=vLLM OpenAI-compatible server (ROCm gfx1150) on :${VLLM_PORT}
After=network.target docker.service
Wants=docker.service

[Service]
Type=simple
WorkingDirectory=/srv/ai/models
Environment=HSA_OVERRIDE_GFX_VERSION=${GFX_VERSION}
Environment=ROCM_PATH=${ROCM_PATH}
Environment=HIP_PATH=${ROCM_PATH}
Environment=LD_LIBRARY_PATH=${ROCM_PATH}/lib:${ROCM_PATH}/lib64:/usr/local/lib
Environment=PATH=${VENV_DIR}/bin:${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=HF_HUB_CACHE=/srv/ai/models/.hf-cache
Environment=HF_HOME=/srv/ai/models/.hf-cache
Environment=PYTHONPATH=/opt/rocm/share/amd_smi:/opt/rocm/lib/python3.12/site-packages
Environment=VLLM_LOGGING_LEVEL=INFO
Environment=VLLM_USE_TVM_FFI=0
Environment=TVM_FFI_DISABLE=1
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Environment=TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
ExecStart=${VENV_DIR}/bin/python -m vllm.entrypoints.openai.api_server --model ${DEFAULT_MODEL_HF} --host 0.0.0.0 --port ${VLLM_PORT} --served-model-name ${DEFAULT_MODEL_NAME} --gpu-memory-utilization 0.70 --max-model-len 4096 --enforce-eager --dtype auto --trust-remote-code --limit-mm-per-prompt='{"image":1,"video":1}' --mm-processor-cache-gb 1 --skip-mm-profiling --max-num-seqs 8
Restart=on-failure
RestartSec=10
User=root
Group=render
SupplementaryGroups=render

[Install]
WantedBy=multi-user.target
UNIT

# --- 5. Open WebUI (docker) ---
echo "[5/6] Creating Open WebUI service (${WEBUI_SERVICE}) on :${WEBUI_PORT} -> vLLM :${VLLM_PORT}..."
cat > "${WEBUI_SERVICE}" << UNIT
[Unit]
Description=Open WebUI (chat UI) for vLLM on :${WEBUI_PORT}
After=network.target vllm.service docker.service
Requires=docker.service
Wants=vllm.service

[Service]
Type=simple
# Pull image on first start
ExecStartPre=-/usr/bin/docker pull ghcr.io/open-webui/open-webui:main
# Run with host networking so UI can reach vLLM at 127.0.0.1:${VLLM_PORT} — privileged LXC needs apparmor=unconfined or docker-default fails (exit 125)
ExecStart=/usr/bin/docker run --rm --name open-webui --network host --security-opt apparmor=unconfined -v open-webui:/app/backend/data -e PORT=${WEBUI_PORT} -e OPENAI_API_BASE_URL=http://127.0.0.1:${VLLM_PORT}/v1 -e BYPASS_MODEL_ACCESS_CONTROL=true ghcr.io/open-webui/open-webui:main
ExecStop=/usr/bin/docker stop open-webui
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT

# Helper switch script for vLLM model
cat > "${SWITCH_SCRIPT}" << 'EOS'
#!/usr/bin/env bash
# vllm-switch-model.sh — switch vLLM model (rewrites ExecStart, restarts)
set -euo pipefail
SERVICE="vllm"
UNIT="/etc/systemd/system/${SERVICE}.service"
echo "Current vLLM model: $(grep -- '--model' "$UNIT" | head -1)"
echo "Available HF models in cache/hub: check https://huggingface.co/models"
read -rp "Enter new HF model ID (e.g. Qwen/Qwen2.5-Coder-32B-Instruct): " NEW_MODEL
if [ -z "$NEW_MODEL" ]; then echo "Aborted."; exit 1; fi
read -rp "Enter served name [default: $(basename "$NEW_MODEL" | tr '[:upper:]' '[:lower:]')]: " SERVED
SERVED="${SERVED:-$(basename "$NEW_MODEL" | tr '[:upper:]' '[:lower:]')}"
# Backup and rewrite ExecStart line
cp "$UNIT" "${UNIT}.backup.$(date +%s)"
# Replace --model and --served-model-name
sed -i -E "s|--model [^ ]+|--model ${NEW_MODEL}|" "$UNIT"
if grep -q -- "--served-model-name" "$UNIT"; then
  sed -i -E "s|--served-model-name [^ ]+|--served-model-name ${SERVED}|" "$UNIT"
else
  sed -i "s|ExecStart=.*vllm.*|& --served-model-name ${SERVED}|" "$UNIT"
fi
systemctl daemon-reload
systemctl restart "$SERVICE"
echo "Waiting for vLLM health on :8000 ..."
for i in {1..90}; do
  if curl -fsS -m 3 -o /dev/null http://127.0.0.1:8000/health 2>/dev/null; then
    echo "[✓] vLLM now serving $NEW_MODEL as $SERVED"
    curl -s http://127.0.0.1:8000/v1/models | head -100 || true
    exit 0
  fi
  sleep 2
done
echo "WARNING: vLLM did not become healthy; check journalctl -u vllm -f"
EOS
chmod +x "${SWITCH_SCRIPT}"
cp "${SWITCH_SCRIPT}" "${MODEL_DIR}/vllm-switch-model.sh" 2>&1 || true

# --- 6. ENABLE & START ---
echo "[6/6] Enabling and starting vLLM + Open WebUI..."
systemctl daemon-reload
systemctl enable --now vllm 2>&1 || true
sleep 5
systemctl enable --now open-webui 2>&1 || true

# --- 7. VERIFICATION ---
echo "[7/7] Verifying..."
echo "[rocm-smi]"
rocm-smi 2>&1 | head -60 || true
echo "[hipconfig]"
hipconfig --version 2>&1 | head -5 || true
echo "[vLLM version]"
"${VENV_DIR}/bin/python" -m vllm --help 2>&1 | head -20 || true
echo "[docker open-webui]"
docker ps 2>&1 | head -20 || true
echo ""
systemctl status vllm --no-pager 2>&1 | tail -40 || true
systemctl status open-webui --no-pager 2>&1 | tail -40 || true
echo ""
echo "[Bootstrap complete - vllm 0.1.0]"
echo "  vLLM API (OpenAI) : http://<container-ip>:${VLLM_PORT}/v1  (health http://<container-ip>:${VLLM_PORT}/health)"
echo "  Open WebUI        : http://<container-ip>:${WEBUI_PORT} (chat UI, BYPASS_MODEL_ACCESS_CONTROL=true)"
echo "  Switch model      : vllm-switch-model.sh  (or ${MODEL_DIR}/vllm-switch-model.sh)"
echo "  GPU               : gfx1150 (890M) HSA_OVERRIDE_GFX_VERSION=${GFX_VERSION} ROCm ${ROCM_VERSION}"
echo "  Model dir         : ${MODEL_DIR} (host <-> LXC) — HF cache at ${MODEL_DIR}/.hf-cache"
echo "  Logs              : journalctl -u vllm -f  |  journalctl -u open-webui -f  |  docker logs -f open-webui"
