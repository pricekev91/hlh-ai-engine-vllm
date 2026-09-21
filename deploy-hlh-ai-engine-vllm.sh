#!/usr/bin/env bash
# deploy-hlh-ai-engine-vllm.sh — 2-file KISS, creates LXC
# Software Bill of Materials — auto-resolved at deploy time (KISS, tok/s first):
#   vLLM latest stable + matching ROCm variant from https://wheels.vllm.ai/rocm/
#   Today: vLLM 0.29.0 + rocm723 (ROCm 7.2.3) — when 0.30.0 ships with rocm1000, it auto-picks 10.0.0.
#   Native installs in LXC via uv (no docker): vllm + open-webui. Host provides amdgpu kernel driver.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_SCRIPT="${SCRIPT_DIR}/configure-hlh-ai-engine-vllm.sh"

usage() {
	cat <<'EOF'
Usage:
	./deploy-hlh-ai-engine-vllm.sh [--update] [--destroy]

This is the direct Proxmox bootstrap path (pure bash, 2-file KISS):
	1) Resolve latest stable vLLM + matching ROCm variant from wheels.vllm.ai (single source of truth)
	2) Create privileged LXC 113 (hlh-ai-engine-vllm) with host ROCm check (ensures host has what vLLM needs)
	3) Configure GPU passthrough (890M gfx1150 Strix Point only)
	4) Start container
	5) Push/run configure-hlh-ai-engine-vllm.sh inside LXC (native vLLM + native Open WebUI via uv)
	   vLLM serving /srv/ai/models/Qwen3.5-9B on :8000, WebUI on :80

If LXC 113 already exists, interactive prompt offers:
  y = destroy & recreate from scratch (full rebuild, ~10-20 min)
  n = abort
  u = update in-place: push/run bootstrap inside existing LXC (fast patch, ~2-5 min)

Flags:
  --update    Non-interactive: update in-place if LXC exists (same as answering 'u')
  --destroy   Non-interactive: destroy & recreate if LXC exists (same as answering 'y')
  --help      Show this help

Env overrides (forwarded into LXC bootstrap):
  VLLM_VERSION=x.y.z          Pin vLLM version (default: auto-resolve latest stable from PyPI)
  VLLM_ROCM_VARIANT=rocm723    Pin ROCm variant (default: auto-resolve highest rocm* for that version)
  ROCM_VERSION=x.y.z           Legacy back-compat: if set, derived to VLLM_ROCM_VARIANT (7.2.3->723)
  PYTHON_VERSION=3.12          Python for venvs (wheels are cp312, must be 3.12)

Examples:
  ./deploy-hlh-ai-engine-vllm.sh              # interactive Y/N/U if 113 exists, auto-resolves 0.29.0/rocm723
  ./deploy-hlh-ai-engine-vllm.sh --update     # fast patch path
  ./deploy-hlh-ai-engine-vllm.sh --destroy    # full rebuild
  VLLM_VERSION=0.29.0 VLLM_ROCM_VARIANT=rocm723 ./deploy-hlh-ai-engine-vllm.sh --update   # pin
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
VLLM_MODEL_DIR="/srv/ai/models"
VLLM_DEFAULT_MODEL="Qwen3.6-35B-A3B-GPTQ-Int4"
WEBUI_PORT="80"

WHEELS_BASE="https://wheels.vllm.ai/rocm"
PYPI_JSON="https://pypi.org/pypi/vllm/json"

NONINTERACTIVE_MODE=""
while [[ $# -gt 0 ]]; do
	case "$1" in
		-h|--help)
			usage
			exit 0
			;;
		--update)
			NONINTERACTIVE_MODE="update"
			;;
		--destroy)
			NONINTERACTIVE_MODE="destroy"
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

# --- Auto-resolve vLLM + ROCm variant (single source of truth, same logic as configure) ---
resolve_vllm_version() {
	local ver="${1:-}"
	if [[ -n "$ver" ]]; then
		ver="$(echo "$ver" | tr -d '[:space:]')"
		if curl -fsSL -o /dev/null "${WHEELS_BASE}/${ver}/" 2>/dev/null; then
			echo "$ver"; return 0
		fi
		echo "FATAL: pinned VLLM_VERSION=${ver} has no wheels at ${WHEELS_BASE}/${ver}/" >&2
		exit 1
	fi
	local latest
	latest="$(curl -fsSL "${PYPI_JSON}" 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin)['info']['version'])" 2>/dev/null || true)"
	if [[ -z "$latest" ]]; then
		echo "FATAL: could not fetch latest vLLM version from ${PYPI_JSON}" >&2
		exit 1
	fi
	if curl -fsSL -o /dev/null "${WHEELS_BASE}/${latest}/" 2>/dev/null; then
		echo "$latest"; return 0
	fi
	echo "PyPI latest ${latest} has no wheels, walking releases..." >&2
	local fallback
	fallback="$(curl -fsSL "${PYPI_JSON}" 2>/dev/null | python3 -c "
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
" 2>/dev/null | head -20 || true)"
	while IFS= read -r v; do
		[[ -z "$v" ]] && continue
		if curl -fsSL -o /dev/null "${WHEELS_BASE}/${v}/" 2>/dev/null; then
			echo "$v"; return 0
		fi
	done <<< "$fallback"
	echo "FATAL: no PyPI release has wheels at ${WHEELS_BASE}/" >&2
	exit 1
}

resolve_rocm_variant() {
	local ver="$1" pin="${2:-}"
	if [[ -n "$pin" ]]; then
		pin="$(echo "$pin" | sed -E 's/^rocm//; s/[^0-9]//g')"
		local p="rocm${pin}"
		if curl -fsSL -o /dev/null "${WHEELS_BASE}/${ver}/${p}/" 2>/dev/null; then
			echo "$p"; return 0
		fi
		echo "FATAL: pinned VLLM_ROCM_VARIANT=${p} not found at ${WHEELS_BASE}/${ver}/${p}/" >&2
		exit 1
	fi
	local listing
	listing="$(curl -fsSL "${WHEELS_BASE}/${ver}/" 2>/dev/null || true)"
	local variants
	variants="$(echo "$listing" | grep -oE 'rocm[0-9]+/' | tr -d '/' | sort -u || true)"
	if [[ -z "$variants" ]]; then
		echo "FATAL: no rocm variant found at ${WHEELS_BASE}/${ver}/" >&2
		exit 1
	fi
	echo "$variants" | sed -E 's/rocm//' | sort -n | tail -1 | awk '{print "rocm"$1}'
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

# Back-compat: ROCM_VERSION env -> VLLM_ROCM_VARIANT if user set it and didn't pin variant
if [[ -n "${ROCM_VERSION:-}" && -z "${VLLM_ROCM_VARIANT:-}" ]]; then
	echo "NOTE: ROCM_VERSION=${ROCM_VERSION} is legacy — deriving VLLM_ROCM_VARIANT from it." >&2
	# 7.2.3 -> 723, 10.0.0 -> 1000
	VLLM_ROCM_VARIANT="$(echo "${ROCM_VERSION}" | awk -F. '{printf "%s%s%s", $1, $2, $3}')"
	# For 10.0.0 the above gives 1000 correct; for 7.2.3 gives 723
	if [[ "${ROCM_VERSION}" == "10.0.0" ]]; then VLLM_ROCM_VARIANT="1000"; fi
	# Normalize to rocm prefix
	if [[ "${VLLM_ROCM_VARIANT}" != rocm* ]]; then VLLM_ROCM_VARIANT="rocm${VLLM_ROCM_VARIANT}"; fi
	echo "  Derived VLLM_ROCM_VARIANT=${VLLM_ROCM_VARIANT}" >&2
fi

RESOLVED_VLLM_VERSION="$(resolve_vllm_version "${VLLM_VERSION:-}")"
RESOLVED_VARIANT="$(resolve_rocm_variant "${RESOLVED_VLLM_VERSION}" "${VLLM_ROCM_VARIANT:-}")"
RESOLVED_ROCM_DOTTED="$(variant_to_dotted "${RESOLVED_VARIANT}")"

VLLM_BACKEND="vLLM ${RESOLVED_VLLM_VERSION} ${RESOLVED_VARIANT} (ROCm ${RESOLVED_ROCM_DOTTED}) native + Open WebUI native :${WEBUI_PORT}"

echo "=== hlh-ai-engine-vllm deploy v0.5.0 ==="
echo "  LXC          : ${LXC_ID} (${LXC_NAME}) ${LXC_IP_CONFIG} on ${POOL}"
echo "  vLLM         : ${RESOLVED_VLLM_VERSION} (PyPI latest stable, auto-resolved)"
echo "  ROCm variant : ${RESOLVED_VARIANT} -> ${RESOLVED_ROCM_DOTTED} (from ${WHEELS_BASE}/${RESOLVED_VLLM_VERSION}/${RESOLVED_VARIANT}/)"
echo "  Backend      : ${VLLM_BACKEND}"
echo "  vLLM index   : ${WHEELS_BASE}/${RESOLVED_VLLM_VERSION}/${RESOLVED_VARIANT}/"
echo "  Install cmd  : uv pip install vllm --extra-index-url ${WHEELS_BASE}/${RESOLVED_VLLM_VERSION}/${RESOLVED_VARIANT}/"
echo "  Model dir    : ${MODEL_HOST_DIR} -> ${MODEL_LXC_DIR} (default model: ${VLLM_DEFAULT_MODEL})"
echo "  WebUI        : native at http://192.168.1.13:${WEBUI_PORT}/ (no docker)"
echo "  Note         : sibling LXC 112 (hlh-ai-engine, llama.cpp) shares the iGPU — it is NOT touched by this deploy."
echo ""

# --- Host ROCm / driver check — ensure host has what wheels need ---
# Wheels bundle userspace, but host must have amdgpu kernel driver/firmware behind /dev/kfd.
# If host has stale amdrocm packages (orphan rocm-smi), warn. Confirm host driver version matches resolved ROCm.
get_host_rocm_version() {
	local ver=""
	ver="$(dpkg-query -W -f='${Version}' amdrocm10.0 2>/dev/null | cut -d- -f1)"
	if [[ -z "$ver" ]]; then
		ver="$(dpkg-query -W -f='${Version}' amdrocm-core10.0 2>/dev/null | cut -d- -f1)"
	fi
	if [[ -z "$ver" ]]; then
		ver="$(dpkg-query -W -f='${Version}' amdrocm-base10.0 2>/dev/null | cut -d- -f1)"
	fi
	if [[ -z "$ver" ]]; then
		ver="$(dpkg-query -W -f='${Version}' amdrocm-core 2>/dev/null | cut -d- -f1)"
	fi
	if [[ -z "$ver" ]]; then
		ver="$(dpkg -l 2>/dev/null | awk '/^ii[ ]+amdrocm-core7/{print $3}' | head -1 | cut -d- -f1)"
	fi
	if [[ -z "$ver" ]]; then
		ver="$(dpkg -l 2>/dev/null | awk '/^ii[ ]+amdrocm7\.14/{print $3}' | head -1 | cut -d- -f1)"
	fi
	if [[ -z "$ver" ]]; then
		ver="$(rocm-smi --version 2>/dev/null | grep -oP '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
	fi
	echo "${ver:-unknown}"
}

HOST_ROCM_VERSION="$(get_host_rocm_version)"
HOST_ROCM_MAJOR="$(echo "${HOST_ROCM_VERSION}" | cut -d. -f1)"
REQ_MAJOR="$(echo "${RESOLVED_ROCM_DOTTED}" | cut -d. -f1)"
HOST_ID="$(. /etc/os-release 2>/dev/null; echo "${ID:-}")"
HOST_VER="$(. /etc/os-release 2>/dev/null; echo "${VERSION_ID:-}")"
HOST_CODENAME="$(. /etc/os-release 2>/dev/null; echo "${VERSION_CODENAME:-}")"
HOST_REPO_DIST=""
if [[ "${HOST_ID}" == "debian" && -n "${HOST_VER}" ]]; then
	HOST_REPO_DIST="debian${HOST_VER%%.*}"
elif [[ "${HOST_ID}" == "ubuntu" && -n "${HOST_VER}" ]]; then
	HOST_REPO_DIST="ubuntu${HOST_VER//./}"
else
	HOST_REPO_DIST="${HOST_CODENAME}"
	if grep -qi "trixie" /etc/os-release 2>/dev/null; then HOST_REPO_DIST="debian13"; fi
	if grep -qi "bookworm" /etc/os-release 2>/dev/null; then HOST_REPO_DIST="debian12"; fi
fi
[[ -z "${HOST_REPO_DIST}" ]] && HOST_REPO_DIST="debian13"
if [[ -z "${HOST_CODENAME}" ]]; then HOST_CODENAME="${HOST_REPO_DIST}"; fi

echo "  Host ROCm    : ${HOST_ROCM_VERSION} (host driver)"
echo "  Host OS      : ${HOST_CODENAME} / ${HOST_REPO_DIST} ($(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d= -f2 | tr -d '\"'))"
echo "  Required ROCm: ${RESOLVED_ROCM_DOTTED} (from ${RESOLVED_VARIANT})"
echo ""

should_upgrade_host=false
if [[ "${HOST_ROCM_VERSION}" == "unknown" ]]; then
	if [[ -e /dev/kfd ]]; then
		echo "Host has no amdrocm apt packages but /dev/kfd exists — kernel driver is likely sufficient (wheels bundle userspace)."
		echo "  rocm-smi orphan check: $(rocm-smi --version 2>&1 | head -1 || echo 'no rocm-smi')"
		echo "  Proceeding without host apt upgrade (use HOST_ROCM_SETUP=1 to force install ${RESOLVED_ROCM_DOTTED})."
	else
		echo "WARNING: Could not detect host ROCm version and /dev/kfd missing — host driver may be missing."
		should_upgrade_host=true
	fi
elif [[ "${HOST_ROCM_MAJOR}" != "${REQ_MAJOR}" ]]; then
	echo "NOTE: Host ROCm ${HOST_ROCM_VERSION} != required ${RESOLVED_ROCM_DOTTED} — wheels bundle userspace, kernel driver likely still OK."
	echo "  Host stays on ${HOST_ROCM_VERSION}, LXC will use ${RESOLVED_ROCM_DOTTED} wheels (no downgrade). Use HOST_ROCM_SETUP=1 to force host install of ${RESOLVED_ROCM_DOTTED}."
elif dpkg --compare-versions "${HOST_ROCM_VERSION}" lt "${RESOLVED_ROCM_DOTTED}" 2>/dev/null; then
	echo "NOTE: Host ROCm ${HOST_ROCM_VERSION} < required ${RESOLVED_ROCM_DOTTED} — wheels bundle userspace, skipping auto-upgrade (use HOST_ROCM_SETUP=1 to force)."
fi
# Allow explicit force via HOST_ROCM_SETUP=1 even when we normally skip
if [[ "${HOST_ROCM_SETUP:-}" == "1" && "${HOST_ROCM_VERSION}" != "${RESOLVED_ROCM_DOTTED}" ]]; then
	echo "HOST_ROCM_SETUP=1 — forcing host upgrade check."
	should_upgrade_host=true
fi

# Only prompt if HOST_ROCM_SETUP is not explicitly 0, and we detected need
if [[ "${should_upgrade_host}" == "true" ]]; then
	echo ""
	echo "Host ROCm appears outdated for ${RESOLVED_VARIANT} (${RESOLVED_ROCM_DOTTED})."
	echo "  Current host: ${HOST_ROCM_VERSION} -> required: ${RESOLVED_ROCM_DOTTED} (${RESOLVED_VARIANT})"
	echo "  Wheels bundle userspace, but host provides amdgpu kernel driver/firmware behind /dev/kfd."
	echo "  This will (if you answer y):"
	if [[ "${REQ_MAJOR}" -ge 10 ]] 2>/dev/null; then
		echo "    - Switch host repo to https://stable.repo.amd.com/rocm/core/packages/${HOST_REPO_DIST} for 10.x"
	else
		echo "    - Use https://repo.amd.com/rocm/packages-multi-arch/${HOST_REPO_DIST} (7.x)"
	fi
	echo "    - apt update && apt install amdrocm* host packages (${RESOLVED_ROCM_DOTTED})"
	echo "    - May require reboot if amdgpu DKMS/firmware changes"
	echo ""
	if [[ "${HOST_ROCM_SETUP:-}" == "0" ]]; then
		echo "HOST_ROCM_SETUP=0 — skipping host upgrade prompt (per env)."
	else
		printf 'Upgrade host ROCm to %s now? [y/N] ' "${RESOLVED_ROCM_DOTTED}"
		read -r _ans
		case "${_ans}" in
			y|Y|yes|YES)
				echo "[0/6] Upgrading host ROCm ${HOST_ROCM_VERSION} -> ${RESOLVED_ROCM_DOTTED} ..."
				mkdir -p /etc/apt/keyrings
				if [[ "${REQ_MAJOR}" -ge 10 ]] 2>/dev/null; then
					echo "  Using stable.repo.amd.com for ROCm 10.x (host ${HOST_REPO_DIST} <- ${HOST_CODENAME})"
					wget -qO - https://stable.repo.amd.com/rocm/gpg/packages.gpg | gpg --dearmor | tee /etc/apt/keyrings/amdrocm.gpg > /dev/null
					[[ -s /etc/apt/keyrings/amdrocm.gpg ]] || { echo "ERROR: ROCm GPG keyring empty (gpg missing or network) — apt will reject the repo" >&2; exit 1; }
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
					[[ -s /etc/apt/keyrings/amdrocm.gpg ]] || { echo "ERROR: ROCm GPG keyring empty (gpg missing or network) — apt will reject the repo" >&2; exit 1; }
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
				if ! apt-get update -o Acquire::Check-Valid-Until=false 2>&1 | tee /tmp/rocm-apt-update.log; then
					echo "WARNING: apt update had errors (see /tmp/rocm-apt-update.log) — continuing if stable repo was fetched"
					cat /tmp/rocm-apt-update.log 2>&1 | tail -40 || true
				fi
				if grep -q "404  Not Found" /tmp/rocm-apt-update.log 2>&1 || grep -q "does not have a Release file" /tmp/rocm-apt-update.log 2>&1; then
					echo "ERROR: Host repo has no Release file (check https://stable.repo.amd.com/rocm/core/packages/ for valid dists)" >&2
					echo "Contents of $(cat /etc/apt/sources.list.d/rocm.list 2>&1)" >&2
				fi
				ROCM_MM_HOST="$(echo "${RESOLVED_ROCM_DOTTED}" | cut -d. -f1,2)"
				echo "  Installing host packages for ROCm ${RESOLVED_ROCM_DOTTED} (try amdrocm${ROCM_MM_HOST}-gfx1150, fallback amdrocm${ROCM_MM_HOST}) ..."
				if ! apt-get install -y --no-install-recommends "amdrocm${ROCM_MM_HOST}-gfx1150" "amdrocm-core${ROCM_MM_HOST}-gfx1150" 2>&1; then
					echo "  Per-GPU host package not found, trying generic amdrocm${ROCM_MM_HOST} ..."
					apt-get install -y --no-install-recommends "amdrocm${ROCM_MM_HOST}" "amdrocm-core${ROCM_MM_HOST}" || {
						echo "  Trying generic amdrocm metapackage ..."
						apt-get install -y --no-install-recommends amdrocm || true
					}
				fi
				if ! dkms status 2>&1 | grep -q amdgpu; then
					echo "  amdgpu dkms not found — installing amdgpu-dkms if available ..."
					apt-get install -y --no-install-recommends amdgpu-dkms 2>&1 || true
				fi
				echo "  Host ROCm upgrade done. New host version: $(get_host_rocm_version)"
				echo "  Host rocm-smi:"
				rocm-smi 2>&1 | head -40 || true
				if dmesg 2>&1 | tail -5 | grep -qi "amdgpu.*firmware"; then
					echo "  NOTE: amdgpu firmware may have changed — reboot recommended if LXC still shows 'no gpu node'."
				fi
				echo ""
				;;
			*)
				echo "Skipping host ROCm upgrade — LXC will be built with ${RESOLVED_ROCM_DOTTED} wheels (userspace bundled), but host driver stays on ${HOST_ROCM_VERSION}."
				echo "You can re-run with VLLM_ROCM_VARIANT=${RESOLVED_VARIANT} or ROCM_VERSION=${HOST_ROCM_VERSION} to match host, or set HOST_ROCM_SETUP=1 and answer 'y'."
				echo ""
				;;
		esac
	fi
else
	if [[ "${HOST_ROCM_VERSION}" != "unknown" ]]; then
		echo "Host ROCm ${HOST_ROCM_VERSION} matches required ${RESOLVED_ROCM_DOTTED} (or kernel driver sufficient) — no upgrade needed."
		echo ""
	fi
fi

confirm_existing_lxc_delete() {
	local answer

	if [[ "${NONINTERACTIVE_MODE}" == "destroy" ]]; then
		echo "Non-interactive --destroy: will destroy & recreate LXC ${LXC_ID}"
		return 0
	elif [[ "${NONINTERACTIVE_MODE}" == "update" ]]; then
		echo "Non-interactive --update: will update in-place (no destroy)"
		return 2
	fi

	printf '%s\n' 'Are you sure?  hlh-ai-engine-vllm is already running and deployed!'
	printf '%s\n' '  y = destroy & recreate from scratch (full rebuild, ~10-20 min)'
	printf '%s\n' '  u = update in-place: push/run bootstrap inside existing LXC (fast patch, ~2-5 min)'
	printf '%s\n' '  n = abort'
	printf '%s' 'Choose [y/N/u]: '
	read -r answer

	case "$answer" in
		y|Y|yes|YES|destroy|DESTROY)
			return 0
			;;
		u|U|update|UPDATE)
			return 2
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

LXC_EXISTS=false
UPDATE_IN_PLACE=false
if pct status "${LXC_ID}" >/dev/null 2>&1; then
	LXC_EXISTS=true
	set +e
	confirm_existing_lxc_delete
	_confirm_rc=$?
	set -e
	if [[ ${_confirm_rc} -eq 2 ]]; then
		UPDATE_IN_PLACE=true
		echo "[1/6] Update in-place requested — will NOT destroy LXC ${LXC_ID}, will patch inside existing container."
	elif [[ ${_confirm_rc} -eq 0 ]]; then
		echo "[1/6] Deleting existing LXC ${LXC_ID} so it can be redeployed..."
		pct stop "${LXC_ID}" >/dev/null 2>&1 || true
		pct destroy "${LXC_ID}" >/dev/null 2>&1 || pct delete "${LXC_ID}"
		LXC_EXISTS=false
	fi
fi

if [[ "${UPDATE_IN_PLACE}" == "true" ]]; then
	echo "[2/6] Skipping LXC creation (update mode) — container ${LXC_ID} already exists, preserving GPU passthrough."
	echo "[3/6] Skipping GPU passthrough wiring (already configured in /etc/pve/lxc/${LXC_ID}.conf)."
	echo "[4/6] Ensuring LXC ${LXC_ID} is running..."
	pct status "${LXC_ID}" 2>&1 | head -5 || true
	if ! pct status "${LXC_ID}" 2>&1 | grep -q "running"; then
		echo "  LXC not running — starting..."
		pct start "${LXC_ID}"
		sleep 12
	fi
	# Give fresh network a moment (systemd-networkd/DHCP); configure also waits, but avoid immediate pct exec fail
	for i in $(seq 1 6); do
		if pct exec "${LXC_ID}" -- getent hosts pypi.org >/dev/null 2>&1; then break; fi
		sleep 3
	done
	echo "[5/6] Running in-container bootstrap (update mode — patch vLLM + WebUI in place)..."
	echo "  Forwarding VLLM_VERSION=${RESOLVED_VLLM_VERSION} VLLM_ROCM_VARIANT=${RESOLVED_VARIANT} -> LXC"
	pct exec "${LXC_ID}" -- mkdir -p /root/ai-engine-bootstrap
	pct push "${LXC_ID}" "$BOOTSTRAP_SCRIPT" /root/ai-engine-bootstrap/configure-hlh-ai-engine-vllm.sh --perms 0755
	pct exec "${LXC_ID}" -- env VLLM_VERSION="${RESOLVED_VLLM_VERSION}" VLLM_ROCM_VARIANT="${RESOLVED_VARIANT}" VLLM_DEFAULT_MODEL="${VLLM_DEFAULT_MODEL}" bash /root/ai-engine-bootstrap/configure-hlh-ai-engine-vllm.sh
	echo "[6/6] Update complete. LXC ${LXC_ID} (${LXC_NAME}) patched in place (no recreate)."
	echo "Model storage: ${MODEL_HOST_DIR} (host) <-> ${MODEL_LXC_DIR} (container) on ${POOL}"
	echo "Backend      : ${VLLM_BACKEND} (gfx1150, ROCm HIP native via wheels)"
	echo "vLLM API     : http://192.168.1.13:8000 ( /health /v1/models /v1/chat/completions ) — agentic coding"
	echo "Open WebUI   : http://192.168.1.13:${WEBUI_PORT}/ (chat UI, no port mapping)"
	echo "Runtime config: /etc/vllm.env + /etc/open-webui.env inside LXC"
	echo "Health: curl -s http://192.168.1.13:8000/health && curl -s http://192.168.1.13:8000/v1/models | head -100"
	echo "WebUI:  curl -s http://192.168.1.13:${WEBUI_PORT}/ | head -20"
	echo "Logs:   ssh root@192.168.1.13 'journalctl -u vllm -f; journalctl -u open-webui -f'"
	exit 0
fi

echo "[2/6] Creating privileged Ubuntu LXC (${LXC_ID}, ${LXC_NAME}) on ${POOL} — ${VLLM_BACKEND}..."
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
	--description "vLLM ${RESOLVED_VLLM_VERSION} ${RESOLVED_VARIANT} + Open WebUI :${WEBUI_PORT} native (no docker), serving ${VLLM_DEFAULT_MODEL} (qwen3.5-9b), host ROCm ${RESOLVED_ROCM_DOTTED}, model storage on ${POOL}"

echo "[3/6] Adding GPU/ROCm passthrough devices..."
cat >> "/etc/pve/lxc/${LXC_ID}.conf" <<'LXCCONF'

# GPU passthrough - 890M iGPU only (gfx1150/Strix Halo, 0000:c9:00.0)
# card0 (226:0) + renderD128 (226:128) is the 890M (1002:150e); card1/2 + renderD129/130 are Tesla K80s (10de:102d) via OCuLink — intentionally NOT passed.
lxc.cgroup2.devices.allow: c 226:0 rwm
lxc.cgroup2.devices.allow: c 226:128 rwm
# kfd major is 511 on ROCm 7.x, 234 on ROCm 10.x (both seen on trixie) — allow both for forward compat
lxc.cgroup2.devices.allow: c 511:0 rwm
lxc.cgroup2.devices.allow: c 234:0 rwm
lxc.mount.entry: /dev/dri/card0 dev/dri/card0 none bind,optional,create=file
lxc.mount.entry: /dev/dri/renderD128 dev/dri/renderD128 none bind,optional,create=file
lxc.mount.entry: /dev/kfd dev/kfd none bind,optional,create=file
LXCCONF

echo "[4/6] Starting LXC ${LXC_ID}..."
pct start "${LXC_ID}"
sleep 12
# Wait for LXC network before bootstrap (fresh container DHCP)
for i in $(seq 1 6); do
	if pct exec "${LXC_ID}" -- getent hosts pypi.org >/dev/null 2>&1; then break; fi
	echo "  Waiting for LXC network... ($i/6)"
	sleep 3
done

if [[ "${UPDATE_IN_PLACE:-false}" != "true" ]] && [[ -t 0 ]] && [[ -z "${NONINTERACTIVE_MODE:-}" ]]; then
	echo ""
	echo "Root password for LXC ${LXC_ID} is not set by pct create."
	echo "You currently do: pct enter ${LXC_ID} -> passwd"
	read -rsp "Set root password now? [Y/n] (empty=no, y=set): " _pw_ask; echo
	case "${_pw_ask}" in
		""|n|N|no|NO) echo "Skipping password set — you can still run: pct exec ${LXC_ID} -- passwd";;
		*)
			read -rsp "New root password for ${LXC_ID}: " _pw; echo
			if [[ -z "${_pw}" ]]; then echo "Empty password — skipping."; else
				read -rsp "Confirm root password: " _pw2; echo
				if [[ "${_pw}" != "${_pw2}" ]]; then
					echo "Passwords do not match — skipping. Run manually: pct exec ${LXC_ID} -- passwd" >&2
				else
					if printf "root:%s\n" "${_pw}" | pct exec "${LXC_ID}" -- chpasswd 2>&1; then
						echo "Root password set for LXC ${LXC_ID}."
					else
						echo "Failed to set password — try manually: pct exec ${LXC_ID} -- passwd" >&2
					fi
				fi
			fi
			unset _pw _pw2
			;;
	esac
	unset _pw_ask
elif [[ -n "${LXC_PASSWORD:-}" ]]; then
	echo "Setting root password via LXC_PASSWORD env..."
	printf "root:%s\n" "${LXC_PASSWORD}" | pct exec "${LXC_ID}" -- chpasswd && echo "Root password set via env."
fi

echo "[5/6] Running in-container bootstrap (native vLLM + WebUI)..."
echo "  Forwarding VLLM_VERSION=${RESOLVED_VLLM_VERSION} VLLM_ROCM_VARIANT=${RESOLVED_VARIANT} -> LXC"
pct exec "${LXC_ID}" -- mkdir -p /root/ai-engine-bootstrap
pct push "${LXC_ID}" "$BOOTSTRAP_SCRIPT" /root/ai-engine-bootstrap/configure-hlh-ai-engine-vllm.sh --perms 0755
pct exec "${LXC_ID}" -- env VLLM_VERSION="${RESOLVED_VLLM_VERSION}" VLLM_ROCM_VARIANT="${RESOLVED_VARIANT}" VLLM_DEFAULT_MODEL="${VLLM_DEFAULT_MODEL}" bash /root/ai-engine-bootstrap/configure-hlh-ai-engine-vllm.sh

echo "[6/6] Deployment complete. LXC ${LXC_ID} (${LXC_NAME}) is running."
echo "Model storage: ${MODEL_HOST_DIR} (host) <-> ${MODEL_LXC_DIR} (container) on ${POOL}"
echo "Backend      : ${VLLM_BACKEND} (gfx1150, ROCm HIP native via wheels, no docker — tok/s optimized)"
echo "vLLM API     : http://192.168.1.13:8000 ( /health /v1/models /v1/chat/completions ) — agentic coding"
echo "Open WebUI   : http://192.168.1.13:${WEBUI_PORT}/ (chat UI, native, port 80 no mapping)"
echo "Runtime config: /etc/vllm.env (vLLM) + /etc/open-webui.env (WebUI) inside LXC"
echo "Health: curl -s http://192.168.1.13:8000/health && curl -s http://192.168.1.13:8000/v1/models | head -100"
echo "WebUI:  curl -s http://192.168.1.13:${WEBUI_PORT}/ | head -20"
echo "Logs:   ssh root@192.168.1.13 'journalctl -u vllm -f'  |  ssh root@192.168.1.13 'journalctl -u open-webui -f'"
