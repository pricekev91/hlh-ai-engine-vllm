#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_SCRIPT="${SCRIPT_DIR}/ansible/files/configure-ai-engine-inside-lxc.sh"

usage() {
	cat <<'EOF'
Usage:
	./deploy-hlh-ai-engine-vllm.sh

This is the direct Proxmox bootstrap path (no OpenTofu):
	1) Create privileged LXC 112 (hlh-ai-engine-vllm)
  2) Configure GPU passthrough
  3) Start container
  4) Push/run in-container bootstrap script
EOF
}

LXC_ID=113
LXC_NAME="hlh-ai-engine-vllm"
LXC_HOSTNAME="hlh-ai-engine-vllm"
LXC_IMAGE="local:vztmpl/ubuntu-24.04-standard_24.04-2_amd64.tar.zst"
POOL="RaidZ1-6TB"
MODEL_HOST_DIR="/srv/ai/models"
MODEL_LXC_DIR="/srv/ai/models"
LXC_ROOTFS_SIZE="64"
LXC_MEMORY="49152"
LXC_CORES="12"
LXC_IP_CONFIG="192.168.1.13/24"
LXC_GATEWAY="192.168.1.1"
# ROCm version tracks latest stable — default is latest upstream (10.0.0 2026-08-26); override with env: ROCM_VERSION=7.14.1 ./deploy-hlh-ai-engine-vllm.sh
# Never pinned — deploy always prints the version it will build (see header/footer) and forwards ROCM_VERSION into the LXC.
ROCM_VERSION="${ROCM_VERSION:-10.0.0}"
VLLM_BACKEND="vLLM ROCm (gfx1150) + Open WebUI"

while [[ $# -gt 0 ]]; do
	case "$1" in
		-h|--help)
			usage
			exit 0
			;;
		*)
			echo "ERROR: Unknown option: $1" >&2
			usage
			exit 1
			;;
	esac
	shift
done

command -v pct >/dev/null 2>&1 || { echo "ERROR: pct command not found. Run on Proxmox host." >&2; exit 1; }
[[ -f "$BOOTSTRAP_SCRIPT" ]] || { echo "ERROR: Bootstrap script not found: $BOOTSTRAP_SCRIPT" >&2; exit 1; }

echo "=== hlh-ai-engine-vllm deploy ==="
echo "  LXC          : ${LXC_ID} (${LXC_NAME}) ${LXC_IP_CONFIG} on ${POOL}"
echo "  ROCm version : ${ROCM_VERSION} (override: ROCM_VERSION=x.y.z ./deploy-hlh-ai-engine-vllm.sh)"
echo "  Backend      : ${VLLM_BACKEND} — vLLM ROCm + Open WebUI"
echo "  Model dir    : ${MODEL_HOST_DIR} -> ${MODEL_LXC_DIR}"
echo ""

# --- Host ROCm upgrade (prox01) — 7.14 is old vs 10.0.0 ---
# The Proxmox host driver must match the LXC user-space ROCm major. Host was on
# 7.14.0 (packages-multi-arch/debian13) while LXC 112 wants 10.0.0 (stable.repo.amd.com).
# Mismatch causes inside LXC: rocm-smi 'No GPUs', rocminfo 'Invalid argument', ggml 'no ROCm device'.
# Prompt and upgrade the host first, before LXC creation.
get_host_rocm_version() {
  local ver=""
  # Prefer installed package version (e.g. 7.14.0-3, 10.0.0-4)
  ver="$(dpkg-query -W -f='${Version}' amdrocm-core 2>/dev/null | cut -d- -f1)"
  if [[ -z "$ver" ]]; then
    ver="$(dpkg -l 2>/dev/null | awk '/^ii[ ]+amdrocm-core7/{print $3}' | head -1 | cut -d- -f1)"
  fi
  if [[ -z "$ver" ]]; then
    ver="$(dpkg -l 2>/dev/null | awk '/^ii[ ]+amdrocm7\.14/{print $3}' | head -1 | cut -d- -f1)"
  fi
  # Fallback: rocm-smi lib version
  if [[ -z "$ver" ]]; then
    ver="$(rocm-smi --version 2>/dev/null | grep -oP '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
  fi
  echo "${ver:-unknown}"
}

HOST_ROCM_VERSION="$(get_host_rocm_version)"
HOST_ROCM_MAJOR="$(echo "${HOST_ROCM_VERSION}" | cut -d. -f1)"
REQ_MAJOR="$(echo "${ROCM_VERSION}" | cut -d. -f1)"
# Map host OS to stable repo dist name: stable uses debian13/ubuntu2404 etc, not codename trixie.
# /etc/os-release on PVE trixie is ID=debian VERSION_ID=13 CODENAME=trixie, but stable wants debian13.
HOST_ID="$(. /etc/os-release 2>/dev/null; echo "${ID:-}")"
HOST_VER="$(. /etc/os-release 2>/dev/null; echo "${VERSION_ID:-}")"
HOST_CODENAME="$(. /etc/os-release 2>/dev/null; echo "${VERSION_CODENAME:-}")"
HOST_REPO_DIST=""
if [[ "${HOST_ID}" == "debian" && -n "${HOST_VER}" ]]; then
  # 13 -> debian13, 12 -> debian12 (matches https://stable.repo.amd.com/rocm/core/packages/debian13/ and legacy packages-multi-arch/debian13)
  HOST_REPO_DIST="debian${HOST_VER%%.*}"
elif [[ "${HOST_ID}" == "ubuntu" && -n "${HOST_VER}" ]]; then
  # 24.04 -> ubuntu2404 (matches stable .../ubuntu2404)
  HOST_REPO_DIST="ubuntu${HOST_VER//./}"
else
  # Fallback to codename mapping
  HOST_REPO_DIST="${HOST_CODENAME}"
  if grep -qi "trixie" /etc/os-release 2>/dev/null; then HOST_REPO_DIST="debian13"; fi
  if grep -qi "bookworm" /etc/os-release 2>/dev/null; then HOST_REPO_DIST="debian12"; fi
  if grep -qi "bullseye" /etc/os-release 2>/dev/null; then HOST_REPO_DIST="debian11"; fi
fi
[[ -z "${HOST_REPO_DIST}" ]] && HOST_REPO_DIST="debian13"
# Keep HOST_CODENAME for display, but use HOST_REPO_DIST for repo URL
if [[ -z "${HOST_CODENAME}" ]]; then HOST_CODENAME="${HOST_REPO_DIST}"; fi

echo "  Host ROCm    : ${HOST_ROCM_VERSION} (host driver)"
echo "  Host OS      : ${HOST_CODENAME} / ${HOST_REPO_DIST} ($(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d= -f2 | tr -d '\"'))"
echo ""

should_upgrade_host=false
if [[ "${HOST_ROCM_VERSION}" == "unknown" ]]; then
  echo "WARNING: Could not detect host ROCm version — will attempt to install ${ROCM_VERSION} on host."
  should_upgrade_host=true
elif [[ "${HOST_ROCM_MAJOR}" != "${REQ_MAJOR}" ]]; then
  echo "Host ROCm major ${HOST_ROCM_MAJOR} != requested ${REQ_MAJOR} (${ROCM_VERSION}). LXC needs matching host driver."
  should_upgrade_host=true
elif dpkg --compare-versions "${HOST_ROCM_VERSION}" lt "${ROCM_VERSION}" 2>/dev/null; then
  echo "Host ROCm ${HOST_ROCM_VERSION} < requested ${ROCM_VERSION} — upgrade recommended (7.14 is old vs 10.0.0)."
  should_upgrade_host=true
fi

if [[ "${should_upgrade_host}" == "true" ]]; then
  echo ""
  echo "Host ROCm upgrade required to ${ROCM_VERSION} before LXC will see the GPU."
  echo "  Current host: ${HOST_ROCM_VERSION} -> target: ${ROCM_VERSION}"
  echo "  This will:"
  echo "    - Switch host repo to https://stable.repo.amd.com/rocm/core/packages/${HOST_REPO_DIST} for 10.x"
  echo "      (or https://repo.amd.com/rocm/packages-multi-arch/${HOST_REPO_DIST} for 7.x)"
  echo "    - apt update && apt install amdrocm${REQ_MAJOR:+${ROCM_VERSION%.*}} host packages"
  echo "    - May require reboot if amdgpu DKMS/firmware changes"
  echo ""
  printf 'Upgrade host ROCm to %s now? [y/N] ' "${ROCM_VERSION}"
  read -r _ans
  case "${_ans}" in
    y|Y|yes|YES)
      echo "[0/6] Upgrading host ROCm ${HOST_ROCM_VERSION} -> ${ROCM_VERSION} ..."
      mkdir -p /etc/apt/keyrings
      if [[ "${REQ_MAJOR}" -ge 10 ]] 2>/dev/null; then
        echo "  Using stable.repo.amd.com for ROCm 10.x (host ${HOST_REPO_DIST} <- ${HOST_CODENAME})"
        wget -qO - https://stable.repo.amd.com/rocm/gpg/packages.gpg | gpg --dearmor | tee /etc/apt/keyrings/amdrocm.gpg > /dev/null
        tee /etc/apt/sources.list.d/rocm.list << EOF
deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] https://stable.repo.amd.com/rocm/core/packages/${HOST_REPO_DIST} stable main
EOF
        tee /etc/apt/preferences.d/rocm-pin << 'PIN'
Package: *
Pin: origin stable.repo.amd.com
Pin-Priority: 1001
PIN
      else
        echo "  Using repo.amd.com/packages-multi-arch for ROCm 7.x (host ${HOST_REPO_DIST} <- ${HOST_CODENAME})"
        wget -qO - https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg | gpg --dearmor | tee /etc/apt/keyrings/amdrocm.gpg > /dev/null
        tee /etc/apt/sources.list.d/rocm.list << EOF
deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] https://repo.amd.com/rocm/packages-multi-arch/${HOST_REPO_DIST} stable main
EOF
        tee /etc/apt/preferences.d/rocm-pin << 'PIN'
Package: *
Pin: origin repo.radeon.com
Pin-Priority: 1001
PIN
      fi
      echo 'APT::Key::GPGCommand "/usr/bin/gpg";' > /etc/apt/apt.conf.d/99gpg-override || true
      # Bullseye security is expired on this trixie host (old leftover) — ignore Valid-Until to unblock apt
      if ! apt-get update -o Acquire::Check-Valid-Until=false 2>&1 | tee /tmp/rocm-apt-update.log; then
        echo "WARNING: apt update had errors (see /tmp/rocm-apt-update.log) — continuing if stable repo was fetched"
        cat /tmp/rocm-apt-update.log 2>&1 | tail -40 || true
      fi
      # If stable repo still 404, diagnostic
      if grep -q "404  Not Found" /tmp/rocm-apt-update.log 2>&1 || grep -q "does not have a Release file" /tmp/rocm-apt-update.log 2>&1; then
        echo "ERROR: Host repo https://stable.repo.amd.com/rocm/core/packages/${HOST_REPO_DIST} has no Release file (check https://stable.repo.amd.com/rocm/core/packages/ for valid dists: debian12 debian13 ubuntu2204 ubuntu2404 etc)" >&2
        echo "Contents of $(cat /etc/apt/sources.list.d/rocm.list 2>&1)" >&2
      fi
      ROCM_MM_HOST="$(echo "${ROCM_VERSION}" | cut -d. -f1,2)"
      echo "  Installing host packages for ROCm ${ROCM_VERSION} (try amdrocm${ROCM_MM_HOST}-gfx1150, fallback amdrocm${ROCM_MM_HOST}) ..."
      if ! apt-get install -y --no-install-recommends "amdrocm${ROCM_MM_HOST}-gfx1150" "amdrocm-core${ROCM_MM_HOST}-gfx1150" 2>&1; then
        echo "  Per-GPU host package not found, trying generic amdrocm${ROCM_MM_HOST} ..."
        apt-get install -y --no-install-recommends "amdrocm${ROCM_MM_HOST}" "amdrocm-core${ROCM_MM_HOST}" || {
          echo "  Trying generic amdrocm metapackage ..."
          apt-get install -y --no-install-recommends amdrocm || true
        }
      fi
      # Also ensure amdgpu dkms if needed
      if ! dkms status 2>&1 | grep -q amdgpu; then
        echo "  amdgpu dkms not found — installing amdgpu-dkms if available ..."
        apt-get install -y --no-install-recommends amdgpu-dkms 2>&1 || true
      fi
      echo "  Host ROCm upgrade done. New host version: $(get_host_rocm_version)"
      echo "  Host rocm-smi:"
      rocm-smi 2>&1 | head -40 || true
      # Check if reboot needed (amdgpu/kfd changed)
      if dmesg 2>&1 | tail -5 | grep -qi "amdgpu.*firmware"; then
        echo "  NOTE: amdgpu firmware may have changed — reboot recommended if LXC still shows 'no gpu node'."
      fi
      echo ""
      ;;
    *)
      echo "Skipping host ROCm upgrade — LXC will be built with ${ROCM_VERSION} but may fail with 'no ROCm device' if host stays on ${HOST_ROCM_VERSION}."
      echo "You can re-run with ROCM_VERSION=${HOST_ROCM_VERSION} ./deploy-hlh-ai-engine-vllm.sh to match host, or re-run and answer 'y' to upgrade host."
      echo ""
      ;;
  esac
fi

confirm_existing_lxc_delete() {
	local answer

	printf '%s\n' 'Are you sure?  hlh-ai-engine-vllm is already running and deployed!'
	printf '%s' 'Delete it and redeploy? [y/N] '
	read -r answer

	case "$answer" in
		y|Y|yes|YES)
			return 0
			;;
		*)
			echo "Aborted." >&2
			exit 1
			;;
	esac
}


echo "[1/6] Creating model storage directory on ${POOL}..."
mkdir -p "${MODEL_HOST_DIR}"
chown 0:0 "${MODEL_HOST_DIR}"
chmod 775 "${MODEL_HOST_DIR}"

if pct status "${LXC_ID}" >/dev/null 2>&1; then
	confirm_existing_lxc_delete
	echo "[1/6] Deleting existing LXC ${LXC_ID} so it can be redeployed..."
	pct stop "${LXC_ID}" >/dev/null 2>&1 || true
	pct destroy "${LXC_ID}" >/dev/null 2>&1 || pct delete "${LXC_ID}"
fi

echo "[2/6] Creating privileged Ubuntu LXC (${LXC_ID}, ${LXC_NAME}) on ${POOL} — ROCm ${ROCM_VERSION}, ${VLLM_BACKEND}..."
pct create "${LXC_ID}" "${LXC_IMAGE}" \
	--storage "${POOL}" \
	--rootfs "${LXC_ROOTFS_SIZE}" \
	--hostname "${LXC_HOSTNAME}" \
	--memory "${LXC_MEMORY}" \
	--cores "${LXC_CORES}" \
	--features nesting=1,keyctl=1 \
	--net0 name=eth0,bridge=vmbr0,ip=${LXC_IP_CONFIG},gw=${LXC_GATEWAY} \
	--unprivileged 0 \
	--onboot 1 \
	--mp0 "${MODEL_HOST_DIR},mp=${MODEL_LXC_DIR}" \
	--description "vLLM AI engine ${VLLM_BACKEND} ROCm ${ROCM_VERSION}, model storage on ${POOL} (Qwen3.5-9B qwen3.5-9b)"

echo "[3/6] Adding GPU/ROCm passthrough devices..."
# Only the 890M iGPU (gfx1150): card1 (226:1) + renderD129 (226:129)
# RX 480 eGPU (gfx803) nodes are intentionally excluded so ROCm cannot
# enumerate the unsupported device as GPU 0 and fail the entire init chain.
# KFD is shared (511:0) but ROCm only sees GPUs that have a visible renderD.
cat >> "/etc/pve/lxc/${LXC_ID}.conf" <<'LXCCONF'

# GPU passthrough - 890M iGPU only (gfx1150/Strix Halo, 0000:c9:00.0)
# card0 (226:0) + renderD128 (226:128) is the 890M (1002:150e); card1/2 + renderD129/130 are Tesla K80s (10de:102d) via OCuLink — intentionally NOT passed.
# Earlier configs used card1/renderD129 when K80 was not enumerated as card0; on current host (trixie, 7.0.14-11-pve) 890M is card0.
lxc.cgroup2.devices.allow: c 226:0 rwm
lxc.cgroup2.devices.allow: c 226:128 rwm
# kfd major is 511 on ROCm 7.x, 234 on ROCm 10.x (both seen on trixie) — allow both for forward compat
lxc.cgroup2.devices.allow: c 511:0 rwm
lxc.cgroup2.devices.allow: c 234:0 rwm
# Mount only the 890M nodes; /dev/dri is created automatically by LXC.
# NOTE: Do NOT use 'lxc.mount.entry: none dev/dri ...' — on Proxmox 9.x that
# incorrectly mounts the host root (rpool/ROOT/pve-1) onto /dev/dri inside the
# container (seen as rpool/ROOT/pve-1 /dev/dri zfs in /proc/mounts), breaking
# DRM and causing rocminfo 'Invalid argument' and rocm-smi 'No GPUs'.
lxc.mount.entry: /dev/dri/card0 dev/dri/card0 none bind,optional,create=file
lxc.mount.entry: /dev/dri/renderD128 dev/dri/renderD128 none bind,optional,create=file
lxc.mount.entry: /dev/kfd dev/kfd none bind,optional,create=file
LXCCONF

echo "[4/6] Starting LXC ${LXC_ID}..."
pct start "${LXC_ID}"
sleep 5

echo "[5/6] Running in-container bootstrap (ROCm ${ROCM_VERSION}, ${VLLM_BACKEND})..."
pct exec "${LXC_ID}" -- mkdir -p /root/ai-engine-bootstrap
pct push "${LXC_ID}" "$BOOTSTRAP_SCRIPT" /root/ai-engine-bootstrap/configure-ai-engine-inside-lxc.sh --perms 0755
pct exec "${LXC_ID}" -- env ROCM_VERSION="${ROCM_VERSION}" bash /root/ai-engine-bootstrap/configure-ai-engine-inside-lxc.sh

echo "[6/6] Deployment complete. LXC ${LXC_ID} (${LXC_NAME}) is running."
echo "Model storage: ${MODEL_HOST_DIR} (host) <-> ${MODEL_LXC_DIR} (container) on ${POOL}"
echo "ROCm version : ${ROCM_VERSION} | Backend: ${VLLM_BACKEND} (gfx1150, ROCm HIP)"
echo "vLLM API (OpenAI-compatible) : http://192.168.1.13:8000 ( /health /v1/models /v1/chat/completions )"
echo "Open WebUI                    : http://192.168.1.13:80 (chat UI, proxied to vLLM :8000, --security-opt apparmor=unconfined)"
echo "Health: curl -s http://192.168.1.13:8000/health && curl -s http://192.168.1.13:8000/v1/models | head -100 && curl -s -I http://192.168.1.13:80 | head -5"
