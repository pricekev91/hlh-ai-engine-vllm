#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANSIBLE_DIR="${SCRIPT_DIR}/ansible"
INVENTORY="${ANSIBLE_DIR}/inventories/hlh-ai-engine-vllm.yml"
PLAYBOOK="${ANSIBLE_DIR}/playbooks/hlh-ai-engine-vllm.yml"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
HOST_OVERRIDE=""
OFFLINE=0

usage() {
    cat <<'EOF'
Usage:
  ./configure-hlh-ai-engine-vllm.sh [--host <ip>] [--offline]

Options:
  --host <ip>  Override target host from inventory.
  --offline    Run in offline mode flag for playbook.
  -h, --help   Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            [[ $# -ge 2 ]] || { echo "ERROR: --host requires a value" >&2; exit 1; }
            HOST_OVERRIDE="$2"
            shift
            ;;
        --offline)
            OFFLINE=1
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

[[ -f "$PLAYBOOK" ]] || { echo "ERROR: Playbook not found: $PLAYBOOK" >&2; exit 1; }
[[ -f "$INVENTORY" ]] || { echo "ERROR: Inventory not found: $INVENTORY" >&2; exit 1; }

cd "$ANSIBLE_DIR"

EXTRA_VARS=("hlh_offline=$([[ "$OFFLINE" -eq 1 ]] && echo true || echo false)")
[[ -n "$HOST_OVERRIDE" ]] && EXTRA_VARS+=("ansible_host=${HOST_OVERRIDE}")

ansible-playbook \
    -i "$INVENTORY" \
    "$PLAYBOOK" \
    --private-key "$SSH_KEY" \
    -e "${EXTRA_VARS[*]}"
