#!/usr/bin/env bash
# deploy-hlh-ai-engine-vllm.sh — 2-file KISS, creates LXC
# Software Bill of Materials (pinned for Tesla V100 Volta sm_70):
#   vLLM 0.19.1 — LAST stable PyPI release with a CUDA 12.8 (cu128) torch:
#     vLLM 0.20.0+ pins torch 2.11.0+cu13, and CUDA 13.0 DROPS Volta (sm_70).
#     Driver R590+ also drops Volta. So on the V100 (host driver R580 580.65.06,
#     the last Volta branch) the stack is capped at vLLM 0.19.1 / torch 2.10.0+cu128.
#   torch 2.10.0 (PyPI default build = CUDA 12.8) — bundles the CUDA 12.8 runtime
#     (nvidia-*-cu12 pip packages) inside the wheel; sm_70 kernels included.
#   Host: NVIDIA R580 580.65.06 on proxmox-kernel-6.14.11-9-pve (owned by
#     hlh-ai-engine-egpu — this script only validates it, it does not install it).
#   LXC userspace: libnvidia-compute-580 + nvidia-utils-580 (580.65.06-0ubuntu1).
#   Native installs in LXC via uv (no docker): vllm + open-webui.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_SCRIPT="${SCRIPT_DIR}/configure-hlh-ai-engine-vllm.sh"

usage() {
	cat <<'EOF'
Usage:
	./deploy-hlh-ai-engine-vllm.sh [--skip-host-driver] [--update] [--destroy]

This is the direct Proxmox bootstrap path (pure bash, 2-file KISS):
	1) Validate host NVIDIA R580 580.65.06 driver + V100 (owned by hlh-ai-engine-egpu)
	2) Create privileged LXC 113 (hlh-ai-engine-vllm) at 192.168.1.13
	3) Configure V100 CUDA passthrough (/dev/nvidia0 + nvidiactl + nvidia-uvm*, c5:00.0)
	4) Start container
	5) Push/run configure-hlh-ai-engine-vllm.sh inside LXC (native vLLM 0.19.1 cu128
	   + native Open WebUI via uv, no docker)
	   vLLM serving /srv/ai/models/Qwen1.5-4B-Chat-GPTQ-Int4 on :8000, WebUI on :80

If LXC 113 already exists, interactive prompt offers:
  y = destroy & recreate from scratch (full rebuild, ~10-20 min)
  n = abort
  u = update in-place: push/run bootstrap inside existing LXC (fast patch, ~2-5 min)

Flags:
  --skip-host-driver   Skip host NVIDIA driver/V100 validation (use after first deploy)
  --update             Non-interactive: update in-place if LXC exists (same as 'u')
  --destroy            Non-interactive: destroy & recreate if LXC exists (same as 'y')
  --help               Show this help

Env overrides (forwarded into LXC bootstrap):
  VLLM_VERSION=x.y.z   Pin vLLM (default 0.19.1 = last stable with cu128/sm_70).
                       WARNING: 0.20.0+ pulls torch 2.11+cu13 which dropped Volta.
  NVIDIA_DRIVER_VERSION  Host driver to expect (default 580.65.06, R580 last for Volta)
  KEEP_111=1           Never stop LXC 111 (hlh-ai-engine-egpu) on the shared V100.
                       Default: deploy stops 111's ai-engine only if its VRAM leaves
                       less than vLLM's budget free (default util 0.30 = ~10GB, so
                       111's ~20GB llama.cpp load coexists with the 4B default model).

Examples:
  ./deploy-hlh-ai-engine-vllm.sh                    # full deploy
  ./deploy-hlh-ai-engine-vllm.sh --skip-host-driver # skip host driver re-validation
  ./deploy-hlh-ai-engine-vllm.sh --update           # fast patch path
  ./deploy-hlh-ai-engine-vllm.sh --destroy          # full rebuild
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
VLLM_DEFAULT_MODEL="Qwen1.5-4B-Chat-GPTQ-Int4"
WEBUI_PORT="80"

# --- PINNED STACK (V100 Volta cc 7.0 — see SBOM header) ---
VLLM_VERSION="${VLLM_VERSION:-0.19.1}"
NVIDIA_DRIVER_VERSION="${NVIDIA_DRIVER_VERSION:-580.65.06}"
GPU_MEM_UTIL="${AI_GPU_MEM_UTIL:-0.30}"        # co-tenancy default (see configure script)

NONINTERACTIVE_MODE=""
SKIP_HOST_DRIVER=false
while [[ $# -gt 0 ]]; do
	case "$1" in
		-h|--help)
			usage
			exit 0
			;;
		--skip-host-driver)
			SKIP_HOST_DRIVER=true
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

if [[ "${VLLM_VERSION}" != "0.19.1" ]]; then
	echo "NOTE: VLLM_VERSION=${VLLM_VERSION} is not the pinned 0.19.1." >&2
	echo "  vLLM 0.20.0+ pins torch 2.11+ (CUDA 13) which DROPS Volta sm_70 — the V100" >&2
	echo "  will not run those builds. Only override with a known sm_70 build." >&2
fi

echo "=== hlh-ai-engine-vllm deploy v0.6.4 ==="
echo "  LXC          : ${LXC_ID} (${LXC_NAME}) ${LXC_IP_CONFIG} on ${POOL}"
echo "  vLLM         : ${VLLM_VERSION} (PyPI CUDA build — last stable with cu128/sm_70)"
echo "  torch        : 2.10.0+cu128 (pulled by vLLM; bundles CUDA 12.8 runtime)"
echo "  GPU          : Tesla V100 GV100GL 32GB (Volta cc 7.0) via OCuLink c5:00.0"
echo "  Host driver  : R580 ${NVIDIA_DRIVER_VERSION} (last branch for Volta)"
echo "  Model dir    : ${MODEL_HOST_DIR} -> ${MODEL_LXC_DIR} (default model: ${VLLM_DEFAULT_MODEL}, GPU util: ${GPU_MEM_UTIL})"
echo "  WebUI        : native at http://192.168.1.13:${WEBUI_PORT}/ (no docker)"
echo "  Note         : the V100 is also passed through to LXC 111 (hlh-ai-engine-egpu, llama.cpp)."
echo "                 VRAM (32GB HBM2) is shared — see README 'GPU co-tenancy'."
echo ""

# --- [0/6] Host NVIDIA driver + V100 validation (driver owned by hlh-ai-engine-egpu) ---
if [[ "$SKIP_HOST_DRIVER" == "false" ]]; then
	echo "[0/6] Host NVIDIA driver check (require R580 ${NVIDIA_DRIVER_VERSION} — last for Volta V100)..."
	DRV="$(modinfo nvidia 2>/dev/null | awk '/^version:/{print $2}')"
	if [[ -z "$DRV" ]]; then
		echo "FATAL: host nvidia driver not loaded." >&2
		echo "  Fix: deploy the sibling repo first (it owns the host driver install):" >&2
		echo "    cd ~/git/hlh-ai-engine-egpu && ./deploy-hlh-ai-engine-egpu.sh" >&2
		echo "  (R580 580.65.06 via .run --dkms on proxmox-kernel-6.14.11-9-pve)," >&2
		echo "  or re-run with --skip-host-driver to bypass." >&2
		exit 1
	fi
	if [[ ! "$DRV" =~ ^580\. ]]; then
		echo "FATAL: host nvidia driver is ${DRV}, need R580 branch (580.x) for the V100." >&2
		echo "  R590+ DROPS Volta; R550 is below the CUDA 12.8 driver floor (570.86.15)." >&2
		echo "  Fix: cd ~/git/hlh-ai-engine-egpu && ./deploy-hlh-ai-engine-egpu.sh" >&2
		echo "  (pins 580.65.06 on 6.14 LTS kernel), then re-run; or --skip-host-driver." >&2
		exit 1
	fi
	KERNEL_VER="$(uname -r)"
	if [[ "$KERNEL_VER" != *6.14* ]]; then
		echo "WARNING: host kernel is ${KERNEL_VER}; validated pin is proxmox-kernel-6.14.11-9-pve" >&2
		echo "  (6.17/7.0 pve kernels break the 550/580 closed DKMS — see hlh-ai-engine-egpu README)." >&2
	fi
	echo "  Host driver ${DRV} on ${KERNEL_VER}"
	set +o pipefail
	nvidia-smi 2>&1 | head -12 || true
	set -o pipefail
	# NOTE: this GV100GL board reports its VBIOS product name "Tesla PG500-216"
	# in nvidia-smi (NOT "Tesla V100") — see hlh-ai-engine-egpu README/CHANGELOG.
	# Gate on GPU count; the in-LXC torch capability check (cc 7.0) is authoritative.
	HOST_GPU_COUNT="$(nvidia-smi -L 2>/dev/null | grep -c 'GPU [0-9]:' || true)"
	if [[ "${HOST_GPU_COUNT}" -ne 1 ]]; then
		echo "FATAL: expected exactly 1 NVIDIA GPU on the host, found ${HOST_GPU_COUNT}." >&2
		echo "  Is the OCuLink eGPU seated (c5:00.0)?" >&2
		nvidia-smi -L 2>&1 | head -5 || true
		exit 1
	fi
	GPU_NAME="$(nvidia-smi -L 2>/dev/null | head -1 | cut -d: -f2- | sed 's/ *(UUID:.*//; s/^ *//; s/ *$//' || true)"
	echo "  GPU: ${GPU_NAME}"
	if ! printf '%s\n' "${GPU_NAME}" | grep -qiE 'V100|PG500|GV100'; then
		echo "WARNING: GPU name '${GPU_NAME}' is not the expected GV100 board name (V100/PG500)." >&2
		echo "  The in-LXC torch check (compute capability 7.0) is the authoritative gate." >&2
	fi
	if command -v lspci >/dev/null 2>&1; then
		NV_PCI="$(lspci -nn 2>/dev/null | awk '/NVIDIA/{print $NF}' | tr -d '[]' | head -1 || true)"
		[[ -n "${NV_PCI}" ]] && echo "  PCI: ${NV_PCI} (expected 10de:1df0 = GV100GL PG500-216)"
	fi

	# Ensure device nodes exist (idempotent — hlh-ai-engine-egpu installs the
	# nvidia-uvm-devices.service for persistence; this covers fresh reboots).
	/sbin/modprobe nvidia 2>/dev/null || true
	/sbin/modprobe nvidia_uvm 2>/dev/null || true
	/sbin/modprobe nvidia_modeset 2>/dev/null || true
	/usr/bin/nvidia-modprobe -u -c 0 2>/dev/null || true
	UVM_MAJOR="$(grep -m1 nvidia-uvm /proc/devices 2>/dev/null | awk '{print $1}')"
	[[ -n "$UVM_MAJOR" ]] || UVM_MAJOR=511
	[ -c /dev/nvidia-uvm ] || mknod -m 666 /dev/nvidia-uvm c "$UVM_MAJOR" 0 2>/dev/null || true
	[ -c /dev/nvidia-uvm-tools ] || mknod -m 666 /dev/nvidia-uvm-tools c "$UVM_MAJOR" 1 2>/dev/null || true
	[ -c /dev/nvidia-modeset ] || mknod -m 666 /dev/nvidia-modeset c 195 254 2>/dev/null || true
	chmod 666 /dev/nvidia-uvm /dev/nvidia-uvm-tools 2>/dev/null || true
	ls -l /dev/nvidia* 2>&1 | head -8 || true
else
	echo "[0/6] Skipping host driver check (--skip-host-driver)"
fi

# --- Co-tenancy: V100 is shared with LXC 111 (llama.cpp) — VRAM is shared ---
# The LXC bootstrap FATALs (VRAM preflight) if free VRAM < AI_GPU_MEM_UTIL*32GB.
# At the co-tenancy default (0.30 = ~10GB) 111's ~20GB llama.cpp load fits
# alongside, so 111 is left running. Stop it here only when its VRAM actually
# prevents vLLM's budget from fitting (e.g. util raised to 0.85, or 111 loaded
# a bigger GGUF). KEEP_111=1 = never stop (then lower AI_GPU_MEM_UTIL or use
# SKIP_VRAM_PREFLIGHT=1).
if [[ "${KEEP_111:-0}" != "1" ]] && command -v pct >/dev/null 2>&1 && pct status 111 >/dev/null 2>&1; then
	if pct status 111 2>&1 | grep -qi "running"; then
		if pct exec 111 -- systemctl is-active ai-engine >/dev/null 2>&1; then
			MEM_TOTAL_HOST="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 32768)"
			MEM_USED_HOST="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)"
			NEED_FREE_HOST="$(awk -v t="${MEM_TOTAL_HOST}" -v u="${GPU_MEM_UTIL}" 'BEGIN{printf "%d", t*u}')"
			MEM_FREE_HOST=$(( MEM_TOTAL_HOST - MEM_USED_HOST ))
			# Stop 111 only if its VRAM leaves less than vLLM's budget free.
			if [[ "${MEM_FREE_HOST}" -lt "${NEED_FREE_HOST}" ]]; then
				echo "[1/6] Co-tenancy: LXC 111 ai-engine holds ${MEM_USED_HOST} MiB VRAM on shared V100 (free ${MEM_FREE_HOST} < needed ${NEED_FREE_HOST}) — stopping for vLLM deploy..."
				echo "  (override: KEEP_111=1 to keep 111 running, then lower AI_GPU_MEM_UTIL or use SKIP_VRAM_PREFLIGHT=1)"
				pct exec 111 -- systemctl stop ai-engine 2>&1 || true
				sleep 3
				MEM_AFTER="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "?")"
				echo "  LXC 111 ai-engine stopped. Host VRAM now ${MEM_AFTER} MiB used."
			fi
		fi
	fi
fi

echo "[1/6] Creating model storage directory on ${POOL}..."
mkdir -p "${MODEL_HOST_DIR}"
chown 0:0 "${MODEL_HOST_DIR}"
chmod 775 "${MODEL_HOST_DIR}"

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
	echo "  Forwarding VLLM_VERSION=${VLLM_VERSION} NVIDIA_DRIVER_VERSION=${NVIDIA_DRIVER_VERSION} -> LXC"
	pct exec "${LXC_ID}" -- mkdir -p /root/ai-engine-bootstrap
	pct push "${LXC_ID}" "$BOOTSTRAP_SCRIPT" /root/ai-engine-bootstrap/configure-hlh-ai-engine-vllm.sh --perms 0755
	pct exec "${LXC_ID}" -- env VLLM_VERSION="${VLLM_VERSION}" NVIDIA_DRIVER_VERSION="${NVIDIA_DRIVER_VERSION}" VLLM_DEFAULT_MODEL="${VLLM_DEFAULT_MODEL}" AI_GPU_MEM_UTIL="${GPU_MEM_UTIL}" bash /root/ai-engine-bootstrap/configure-hlh-ai-engine-vllm.sh
	echo "[6/6] Update complete. LXC ${LXC_ID} (${LXC_NAME}) patched in place (no recreate)."
	echo "Model storage: ${MODEL_HOST_DIR} (host) <-> ${MODEL_LXC_DIR} (container) on ${POOL}"
	echo "Backend      : vLLM ${VLLM_VERSION} CUDA 12.8 (V100 GV100 32GB sm_70, native via uv, no docker)"
	echo "vLLM API     : http://192.168.1.13:8000 ( /health /v1/models /v1/chat/completions )"
	echo "Open WebUI   : http://192.168.1.13:${WEBUI_PORT}/ (chat UI, no port mapping)"
	echo "Runtime config: /etc/vllm.env + /etc/open-webui.env inside LXC"
	echo "Health: curl -s http://192.168.1.13:8000/health && curl -s http://192.168.1.13:8000/v1/models | head -100"
	echo "WebUI:  curl -s http://192.168.1.13:${WEBUI_PORT}/ | head -20"
	echo "Logs:   ssh root@192.168.1.13 'journalctl -u vllm -f; journalctl -u open-webui -f'"
	exit 0
fi

echo "[2/6] Creating privileged Ubuntu LXC (${LXC_ID}, ${LXC_NAME}) on ${POOL} — vLLM ${VLLM_VERSION} CUDA 12.8 native (no docker)..."
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
	--description "vLLM ${VLLM_VERSION} CUDA 12.8 + Open WebUI :${WEBUI_PORT} native (no docker), serving ${VLLM_DEFAULT_MODEL} on Tesla V100 GV100 32GB (sm_70) via OCuLink c5:00.0, host driver R580 ${NVIDIA_DRIVER_VERSION}, model storage on ${POOL}"

echo "[3/6] Adding V100 CUDA passthrough (single GV100 32GB + UVM)..."
# V100 presents as single PCI device c5:00.0 (10de:1df0) via OCuLink. LXC passthrough via /dev, not hostpci.
# Host /dev/nvidia* is created by the nvidia driver (R580) after modprobe; expose via cgroup + bind-mount.
# UVM major is dynamic: 507 (470) / 508 (580) / 511 (caps) — allow all. nvidia-modeset is 195:254.
# NOTE: the same /dev/nvidia* nodes are also passed through to LXC 111 (hlh-ai-engine-egpu);
# the NVIDIA driver multiplexes access — VRAM (32GB) is shared, see README.
cat >> "/etc/pve/lxc/${LXC_ID}.conf" <<'LXCCONF'

# V100 Tesla GV100 32GB (cc 7.0 Volta) — CUDA 12.8, driver R580 580.65.06 (last for Volta)
# c5:00.0 (10de:1df0) via OCuLink GPP x4; IOMMU group 20
# Expose single chip as nvidia0 plus control nodes (CUDA)
lxc.cgroup2.devices.allow: c 195:* rwm
lxc.cgroup2.devices.allow: c 507:* rwm
lxc.cgroup2.devices.allow: c 508:* rwm
lxc.cgroup2.devices.allow: c 510:* rwm
lxc.cgroup2.devices.allow: c 511:* rwm
lxc.mount.entry: /dev/nvidia0 dev/nvidia0 none bind,optional,create=file
lxc.mount.entry: /dev/nvidiactl dev/nvidiactl none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm dev/nvidia-uvm none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm-tools dev/nvidia-uvm-tools none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-modeset dev/nvidia-modeset none bind,optional,create=file
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

if [[ -t 0 ]] && [[ -z "${NONINTERACTIVE_MODE:-}" ]]; then
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

echo "[5/6] Running in-container bootstrap (native vLLM CUDA + WebUI)..."
echo "  Forwarding VLLM_VERSION=${VLLM_VERSION} NVIDIA_DRIVER_VERSION=${NVIDIA_DRIVER_VERSION} -> LXC"
pct exec "${LXC_ID}" -- mkdir -p /root/ai-engine-bootstrap
pct push "${LXC_ID}" "$BOOTSTRAP_SCRIPT" /root/ai-engine-bootstrap/configure-hlh-ai-engine-vllm.sh --perms 0755
pct exec "${LXC_ID}" -- env VLLM_VERSION="${VLLM_VERSION}" NVIDIA_DRIVER_VERSION="${NVIDIA_DRIVER_VERSION}" VLLM_DEFAULT_MODEL="${VLLM_DEFAULT_MODEL}" AI_GPU_MEM_UTIL="${GPU_MEM_UTIL}" bash /root/ai-engine-bootstrap/configure-hlh-ai-engine-vllm.sh

echo "[5/6] Verifying bootstrap (fail-fast instead of silent port-8000 refused)..."
pct exec "${LXC_ID}" -- systemctl is-active vllm >/dev/null 2>&1 || { echo "ERROR: vllm.service not active after bootstrap — check: pct exec ${LXC_ID} -- journalctl -u vllm -n 100" >&2; exit 1; }
pct exec "${LXC_ID}" -- curl -fsS -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || echo "WARNING: vllm active but /health not ready yet (model load can take minutes) — check journalctl" >&2

echo "[6/6] Deployment complete. LXC ${LXC_ID} (${LXC_NAME}) is running."
echo "Model storage: ${MODEL_HOST_DIR} (host) <-> ${MODEL_LXC_DIR} (container) on ${POOL}"
echo "Backend      : vLLM ${VLLM_VERSION} CUDA 12.8 (V100 GV100 32GB sm_70, native via uv, no docker)"
echo "vLLM API     : http://192.168.1.13:8000 ( /health /v1/models /v1/chat/completions ) — agentic coding"
echo "Open WebUI   : http://192.168.1.13:${WEBUI_PORT}/ (chat UI, native, port 80 no mapping)"
echo "Runtime config: /etc/vllm.env (vLLM) + /etc/open-webui.env (WebUI) inside LXC"
echo "Health: curl -s http://192.168.1.13:8000/health && curl -s http://192.168.1.13:8000/v1/models | head -100"
echo "WebUI:  curl -s http://192.168.1.13:${WEBUI_PORT}/ | head -20"
echo "Logs:   ssh root@192.168.1.13 'journalctl -u vllm -f'  |  ssh root@192.168.1.13 'journalctl -u open-webui -f'"
echo "GPU:    ssh root@192.168.1.13 'nvidia-smi'"
