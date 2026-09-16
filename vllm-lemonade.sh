#!/usr/bin/env bash
#
# install-vllm-openwebui.sh
#
# One-and-done installer for:
#   1) vLLM on AMD ROCm, via Lemonade Server's prebuilt vllm-rocm bundle
#      (self-contained: its own Python/PyTorch-ROCm/Triton/vLLM — no system
#      ROCm install or version-matching required). gfx1150 (Radeon 890M /
#      Strix Point) and gfx1151 (Strix Halo) are the two validated targets.
#   2) Open WebUI (Docker), pointed at Lemonade's OpenAI-compatible endpoint.
#
# Run this INSIDE the LXC/host that already has /dev/kfd and /dev/dri/card*
# passed through (see your existing deploy-hlh-ai-engine-vllm.sh for that).
#
# Usage:
#   sudo ./install-vllm-openwebui.sh
#   sudo MODEL="Qwen3.5-9B-vLLM" LEMONADE_PORT=8000 OWUI_PORT=3000 ./install-vllm-openwebui.sh
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Config (override via env)
# ---------------------------------------------------------------------------
MODEL="${MODEL:-Qwen3.5-9B-vLLM}"          # Model recipe name Lemonade will pull/serve
LEMONADE_PORT="${LEMONADE_PORT:-8000}"     # vLLM/Lemonade OpenAI-compatible API port
OWUI_PORT="${OWUI_PORT:-3000}"             # Open WebUI port (host)
OWUI_CONTAINER_NAME="${OWUI_CONTAINER_NAME:-open-webui}"

log()  { echo -e "\n=== $* ==="; }
warn() { echo "WARNING: $*" >&2; }

[[ "$(id -u)" -eq 0 ]] || { echo "ERROR: run as root (sudo)." >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0) Sanity checks
# ---------------------------------------------------------------------------
log "0/6 Checking GPU device nodes"
if [[ ! -e /dev/kfd ]]; then
  warn "/dev/kfd not found. If you're in an LXC, this needs to be passed through"
  warn "from the host first (see your Proxmox deploy script's GPU passthrough section)."
fi
ls /dev/dri/ 2>/dev/null || warn "/dev/dri not found — GPU passthrough may be missing."

# ---------------------------------------------------------------------------
# 1) Install Lemonade Server
# ---------------------------------------------------------------------------
log "1/6 Installing Lemonade Server"
if ! command -v lemonade-server >/dev/null 2>&1 && ! command -v lemonade >/dev/null 2>&1; then
  apt-get update -y
  apt-get install -y --no-install-recommends software-properties-common curl ca-certificates
  add-apt-repository -y ppa:lemonade-team/stable
  apt-get update -y
  apt-get install -y lemonade-server
else
  echo "Lemonade already installed, skipping."
fi

LEMONADE_BIN="$(command -v lemonade-server || command -v lemonade)"
echo "Using: ${LEMONADE_BIN}"

# ---------------------------------------------------------------------------
# 2) Install the vLLM ROCm backend (auto-detects gfx target, e.g. gfx1150)
# ---------------------------------------------------------------------------
log "2/6 Installing vLLM ROCm backend (this pulls the per-GPU-target bundle, can take a while)"
"${LEMONADE_BIN}" backends install vllm:rocm || {
  echo "ERROR: vLLM ROCm backend install failed. This backend is EXPERIMENTAL on gfx1150 —" >&2
  echo "check 'https://lemonade-server.ai/docs/guide/configuration/vllm/' for known issues" >&2
  echo "and kernel prerequisites (CWSR sysfs support)." >&2
  exit 1
}

# ---------------------------------------------------------------------------
# 3) Pull the model ahead of time (so the systemd service doesn't stall on first request)
# ---------------------------------------------------------------------------
log "3/6 Pulling model: ${MODEL}"
"${LEMONADE_BIN}" pull "${MODEL}" || warn "Pull failed/unsupported for this Lemonade version — will attempt on first 'run' instead."

# ---------------------------------------------------------------------------
# 4) Run Lemonade/vLLM as a systemd service (persistent, survives reboot)
# ---------------------------------------------------------------------------
log "4/6 Creating systemd service (lemonade-vllm.service)"
LEMONADE_REAL_BIN="$(command -v lemonade-server || command -v lemonade)"

cat > /etc/systemd/system/lemonade-vllm.service <<EOF
[Unit]
Description=Lemonade Server (vLLM ROCm backend)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=${LEMONADE_REAL_BIN} serve --host 0.0.0.0 --port ${LEMONADE_PORT} --backend vllm:rocm --model ${MODEL}
Restart=on-failure
RestartSec=5
Environment=HOME=/root

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now lemonade-vllm.service

log "Waiting for API to come up on :${LEMONADE_PORT}"
for i in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:${LEMONADE_PORT}/api/v1/models" >/dev/null 2>&1; then
    echo "API is up."
    break
  fi
  sleep 2
  if [[ "$i" -eq 30 ]]; then
    warn "API didn't respond after 60s — check: journalctl -u lemonade-vllm -f"
  fi
done

# ---------------------------------------------------------------------------
# 5) Install Docker (if missing) and run Open WebUI
# ---------------------------------------------------------------------------
log "5/6 Installing Docker (if needed) and launching Open WebUI"
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
  systemctl enable --now docker
fi

docker rm -f "${OWUI_CONTAINER_NAME}" >/dev/null 2>&1 || true

# --network=host lets the container reach Lemonade at 127.0.0.1 directly,
# which is the simplest option when both run on the same LXC/host.
docker run -d \
  --name "${OWUI_CONTAINER_NAME}" \
  --network host \
  --restart unless-stopped \
  -e OPENAI_API_BASE_URL="http://127.0.0.1:${LEMONADE_PORT}/api/v1" \
  -e OPENAI_API_KEY="lemonade" \
  -e PORT="${OWUI_PORT}" \
  -e WEBUI_AUTH="false" \
  -v open-webui:/app/backend/data \
  ghcr.io/open-webui/open-webui:main

# ---------------------------------------------------------------------------
# 6) Done
# ---------------------------------------------------------------------------
IP_ADDR="$(hostname -I 2>/dev/null | awk '{print $1}')"
log "6/6 Done"
cat <<EOF

vLLM (via Lemonade)  : http://${IP_ADDR:-<host-ip>}:${LEMONADE_PORT}/api/v1  (OpenAI-compatible)
  - models:  curl http://127.0.0.1:${LEMONADE_PORT}/api/v1/models
  - logs:    journalctl -u lemonade-vllm -f
  - service: systemctl {status,restart,stop} lemonade-vllm

Open WebUI           : http://${IP_ADDR:-<host-ip>}:${OWUI_PORT}
  - Auth is disabled (WEBUI_AUTH=false) for one-shot local use. Set it back on
    if this is reachable beyond localhost: remove that env var and restart
    the container.
  - It should auto-detect the Lemonade connection via OPENAI_API_BASE_URL.
    If not: Admin Settings -> Connections -> OpenAI -> set URL to
    http://127.0.0.1:${LEMONADE_PORT}/api/v1

NOTE: gfx1150 (890M) support in Lemonade's vLLM backend is labeled
experimental upstream. If you hit an OOM despite free memory, a known fix
for gfx1150 APUs is adding 'ttm.pages_limit=12582912' to the host kernel
cmdline (GRUB) and rebooting — that's a host-level change, not something
this script touches.
EOF
