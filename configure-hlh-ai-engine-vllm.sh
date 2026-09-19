#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_SCRIPT="${SCRIPT_DIR}/configure-ai-engine-inside-lxc.sh"
LXC_ID=113
DEFAULT_HOST="192.168.1.13"

usage() {
	cat <<'EOF'
Usage:
  ./configure-hlh-ai-engine-vllm.sh [--host <ip>] [--lxc <id>]

Pure bash reconfiguration:
  Pushes and runs configure-ai-engine-inside-lxc.sh inside existing LXC 113
  via pct push/pct exec (Proxmox host) or ssh (remote).

Options:
  --host <ip>   Target LXC IP for ssh mode (default: 192.168.1.13, used when pct unavailable)
  --lxc <id>    LXC ID (default: 113)
  -h, --help    Show this help.
EOF
}

HOST_OVERRIDE=""
LXC_OVERRIDE=""

while [[ $# -gt 0 ]]; do
	case "$1" in
		--host)
			[[ $# -ge 2 ]] || { echo "ERROR: --host requires a value" >&2; exit 1; }
			HOST_OVERRIDE="$2"
			shift
			;;
		--lxc)
			[[ $# -ge 2 ]] || { echo "ERROR: --lxc requires a value" >&2; exit 1; }
			LXC_OVERRIDE="$2"
			shift
			;;
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

[[ -n "$LXC_OVERRIDE" ]] && LXC_ID="$LXC_OVERRIDE"
TARGET_HOST="${HOST_OVERRIDE:-$DEFAULT_HOST}"

[[ -f "$BOOTSTRAP_SCRIPT" ]] || { echo "ERROR: Bootstrap script not found: $BOOTSTRAP_SCRIPT" >&2; exit 1; }

# Prefer pct (Proxmox host) — pure bash
if command -v pct >/dev/null 2>&1 && pct status "$LXC_ID" >/dev/null 2>&1; then
	echo "=== hlh-ai-engine-vllm configure (pct) ==="
	echo "  LXC: $LXC_ID -> $TARGET_HOST"
	echo "  Bootstrap: $BOOTSTRAP_SCRIPT"
	pct exec "$LXC_ID" -- mkdir -p /root/ai-engine-bootstrap
	pct push "$LXC_ID" "$BOOTSTRAP_SCRIPT" /root/ai-engine-bootstrap/configure-ai-engine-inside-lxc.sh --perms 0755
	# Forward ROCM_VERSION if set in caller env
	if [[ -n "${ROCM_VERSION:-}" ]]; then
		pct exec "$LXC_ID" -- env ROCM_VERSION="$ROCM_VERSION" bash /root/ai-engine-bootstrap/configure-ai-engine-inside-lxc.sh
	else
		pct exec "$LXC_ID" -- bash /root/ai-engine-bootstrap/configure-ai-engine-inside-lxc.sh
	fi
	echo "Configure complete via pct."
	exit 0
fi

# Fallback: ssh to LXC IP
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
echo "=== hlh-ai-engine-vllm configure (ssh) ==="
echo "  Host: $TARGET_HOST (LXC $LXC_ID)"
echo "  Bootstrap: $BOOTSTRAP_SCRIPT"
if [[ -f "$SSH_KEY" ]]; then
	SSH_OPTS=(-o StrictHostKeyChecking=no -i "$SSH_KEY")
else
	SSH_OPTS=(-o StrictHostKeyChecking=no)
fi
# Push via scp then exec
scp "${SSH_OPTS[@]}" "$BOOTSTRAP_SCRIPT" "root@${TARGET_HOST}:/tmp/configure-ai-engine-inside-lxc.sh"
# shellcheck disable=SC2029
ssh "${SSH_OPTS[@]}" "root@${TARGET_HOST}" "chmod +x /tmp/configure-ai-engine-inside-lxc.sh && env ROCM_VERSION=\"${ROCM_VERSION:-10.0.0}\" bash /tmp/configure-ai-engine-inside-lxc.sh"
echo "Configure complete via ssh."
