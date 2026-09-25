#!/usr/bin/env bash
# dl-hf.sh — download a Hugging Face model into a folder in this dir
# (/srv/ai/models/<model>) so vLLM can serve it directly.
#
# Uses the `hf` CLI (huggingface_hub; replaces the deprecated
# huggingface-cli). Installs it if missing:
#   curl -LsSf https://hf.co/cli/install.sh | bash -s
#
# Usage: ./dl-hf.sh <hf-model-id> [extra hf download flags]
#   ./dl-hf.sh Qwen/Qwen1.5-4B-Chat-GPTQ-Int4
#   ./dl-hf.sh Qwen/Qwen3.6-35B-A3B-GPTQ-Int4 --include '*.json'
#
# HF_TOKEN: set it in the environment for gated/private repos.
# Not needed for public repos.
set -euo pipefail

MODEL_ID="${1:-}"
[[ -n "$MODEL_ID" ]] || { echo "Usage: $0 <hf-model-id> [extra hf download flags]" >&2; exit 1; }
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="${MODEL_ID##*/}"
[[ "$MODEL_NAME" == "$MODEL_ID" ]] && MODEL_NAME="model"
DEST_DIR="${SCRIPT_DIR}/${MODEL_NAME}"

# Ensure the `hf` CLI is available (installer may drop it in ~/.local/bin).
export PATH="$HOME/.local/bin:$HOME/.hf/bin:$PATH"
if ! command -v hf >/dev/null 2>&1; then
	echo "hf CLI not found — installing (curl -LsSf https://hf.co/cli/install.sh | bash -s)"
	curl -LsSf https://hf.co/cli/install.sh | bash -s
fi
command -v hf >/dev/null 2>&1 || { echo "ERROR: hf CLI unavailable (install failed?)" >&2; exit 1; }

mkdir -p "$DEST_DIR"
echo "Downloading ${MODEL_ID} -> ${DEST_DIR}"
hf download "$MODEL_ID" --local-dir "$DEST_DIR" "$@"

echo
echo "=== done: ${DEST_DIR} ==="
ls -lh "$DEST_DIR"
du -sh "$DEST_DIR"
