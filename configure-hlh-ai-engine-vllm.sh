#!/usr/bin/env bash
# configure-hlh-ai-engine-vllm.sh
# Version: 0.5.1
# Description: Native vLLM + native Open WebUI on Ubuntu 24.04 LXC via wheels.vllm.ai
#              Target: Radeon 890M (gfx1150 / Strix Point). No Docker — tok/s first.
#
# Design rules (v0.5.0 simplicity):
#   1. Single source of truth: uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/<ver>/<variant>
#      Starting point: uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/0.29.0/rocm723 (today's stable)
#      Auto-resolves latest stable at deploy time — zero script changes when 0.30.0 ships.
#   2. No Docker. Both vLLM and Open WebUI run native via uv venvs in the same LXC (shared GPU, shared /srv/ai/models).
#   3. ROCm runtime userspace (MIOpen/rocBLAS/roctracer/HIP runtime) comes from the AMD apt repo
#      matching the wheel's ROCm version — torch wheels do NOT bundle these libs.
#      Host kernel amdgpu driver behind /dev/kfd is the only host requirement.
#   4. Any failed check aborts loudly. No shims/fallbacks silently hiding a broken env.
#   5. Two venvs: /opt/vllm-venv (vLLM) and /opt/open-webui-venv (WebUI) — avoids dep conflicts.

set -euo pipefail

# --- CONFIGURABLE (env-overridable) ---
VLLM_VERSION="${VLLM_VERSION:-}"               # pin, e.g. 0.29.0; else auto from PyPI
VLLM_ROCM_VARIANT="${VLLM_ROCM_VARIANT:-}"     # pin, e.g. rocm723 or 723; else auto
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"       # wheels are cp312 (manylinux_2_39); 3.14 will fail
VENV_DIR="${VENV_DIR:-/opt/vllm-venv}"
WEBUI_VENV_DIR="${WEBUI_VENV_DIR:-/opt/open-webui-venv}"

AI_PORT="${AI_PORT:-8000}"
WEBUI_PORT="${WEBUI_PORT:-80}"
MODEL_DIR="${MODEL_DIR:-/srv/ai/models}"
DEFAULT_MODEL_PATH="${DEFAULT_MODEL_PATH:-${MODEL_DIR}/Qwen3.6-35B-A3B-GPTQ-Int4}"
DEFAULT_MODEL_NAME="${DEFAULT_MODEL_NAME:-qwen3.6-35b-a3b-gptq-int4}"
GPU_MEM_UTIL="${AI_GPU_MEM_UTIL:-0.60}"
MAX_MODEL_LEN="${AI_MAX_MODEL_LEN:-16384}"
ENABLE_ROOT_PASSWORD_SSH="${ENABLE_ROOT_PASSWORD_SSH:-1}"

VLLM_SERVICE="/etc/systemd/system/vllm.service"
WEBUI_SERVICE="/etc/systemd/system/open-webui.service"
RUNNER="/usr/local/bin/vllm-run.sh"
WEBUI_RUNNER="/usr/local/bin/open-webui-run.sh"
ENV_FILE="/etc/vllm.env"
WEBUI_ENV_FILE="/etc/open-webui.env"
SWITCH_SCRIPT="/usr/local/bin/vllm-switch-model.sh"

WHEELS_BASE="https://wheels.vllm.ai/rocm"
PYPI_JSON="https://pypi.org/pypi/vllm/json"

# --- helpers: resolve latest stable + variant + python ---
# All helpers are network-resilient: retry with backoff and trust pinned values
# when deploy already resolved them (fresh LXC network may need ~10s after pct start).
curl_retry() {
  local url="$1" tries=6 delay=3
  for i in $(seq 1 $tries); do
    if curl -fsSL --connect-timeout 10 --max-time 20 -o /dev/null "$url" 2>/dev/null; then
      return 0
    fi
    sleep $delay
  done
  return 1
}
curl_fetch() {
  local url="$1" tries=6 delay=3
  for i in $(seq 1 $tries); do
    if curl -fsSL --connect-timeout 10 --max-time 20 "$url" 2>/dev/null; then
      return 0
    fi
    sleep $delay
  done
  return 1
}

resolve_vllm_version() {
  local ver="$1"
  if [[ -n "$ver" ]]; then
    ver="$(echo "$ver" | tr -d '[:space:]')"
    if curl_retry "${WHEELS_BASE}/${ver}/"; then
      echo "$ver"; return 0
    fi
    echo "WARNING: pinned VLLM_VERSION=${ver} probe failed for ${WHEELS_BASE}/${ver}/ — trusting pinned value (deploy already resolved, LXC network may be warming up)" >&2
    echo "$ver"; return 0
  fi
  local latest=""
  latest="$(curl_fetch "${PYPI_JSON}" | python3 -c "import sys,json;print(json.load(sys.stdin)['info']['version'])" 2>/dev/null || true)"
  if [[ -z "$latest" ]]; then
    echo "FATAL: could not fetch latest vLLM version from ${PYPI_JSON} (network down?)" >&2
    exit 1
  fi
  if curl_retry "${WHEELS_BASE}/${latest}/"; then
    echo "$latest"; return 0
  fi
  echo "PyPI latest ${latest} has no wheels, walking releases..." >&2
  local fallback
  fallback="$(curl_fetch "${PYPI_JSON}" | python3 -c "
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
    if curl_retry "${WHEELS_BASE}/${v}/"; then
      echo "$v"; return 0
    fi
  done <<< "$fallback"
  echo "FATAL: no PyPI release has wheels at ${WHEELS_BASE}/" >&2
  exit 1
}

resolve_rocm_variant() {
  local ver="$1" pin="$2"
  if [[ -n "$pin" ]]; then
    pin="$(echo "$pin" | sed -E 's/^rocm//; s/[^0-9]//g')"
    local p="rocm${pin}"
    if curl_retry "${WHEELS_BASE}/${ver}/${p}/"; then
      echo "$p"; return 0
    fi
    echo "WARNING: pinned VLLM_ROCM_VARIANT=${p} probe failed for ${WHEELS_BASE}/${ver}/${p}/ — trusting pinned value" >&2
    echo "$p"; return 0
  fi
  local listing
  listing="$(curl_fetch "${WHEELS_BASE}/${ver}/" || true)"
  local variants
  variants="$(echo "$listing" | grep -oE 'rocm[0-9]+/' | tr -d '/' | sort -u || true)"
  if [[ -z "$variants" ]]; then
    echo "FATAL: no rocm variant found at ${WHEELS_BASE}/${ver}/" >&2
    exit 1
  fi
  echo "$variants" | sed -E 's/rocm//' | sort -n | tail -1 | awk '{print "rocm"$1}'
}

resolve_python_version() {
  local ver="$1" variant="$2"
  local listing
  listing="$(curl_fetch "${WHEELS_BASE}/${ver}/${variant}/vllm/" || true)"
  local whl
  whl="$(echo "$listing" | grep -oE 'vllm-[^"]+\.whl' | grep -vE 'rc|\.dev|dev[0-9]' | head -1 || true)"
  if [[ -z "$whl" ]]; then
    whl="$(echo "$listing" | grep -oE 'vllm-[^"]+\.whl' | head -1 || true)"
  fi
  if [[ -z "$whl" ]]; then
    echo "WARNING: could not list vllm wheel at ${WHEELS_BASE}/${ver}/${variant}/vllm/, defaulting to ${PYTHON_VERSION}" >&2
    echo "${PYTHON_VERSION}"; return 0
  fi
  local cp
  cp="$(echo "$whl" | grep -oE 'cp[0-9]+' | head -1 || true)"
  if [[ -n "$cp" ]]; then
    local num="${cp#cp}"
    echo "${num:0:1}.${num:1}"
    return 0
  fi
  echo "${PYTHON_VERSION}"
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

# --- 0. PRE-FLIGHT ---
echo "[0/8] Pre-flight checks..."
if [[ ! -e /dev/kfd ]]; then
  echo "FATAL: /dev/kfd missing. Fix LXC GPU passthrough (/dev/kfd + /dev/dri) first." >&2
  exit 1
fi
ls -l /dev/kfd /dev/dri/ 2>&1 | head -10 || true

systemctl stop vllm 2>/dev/null || true
systemctl stop open-webui 2>/dev/null || true

# --- 1. BASE PACKAGES (do first so curl + network are ready before resolve) ---
echo "[1/8] Installing base packages..."
export DEBIAN_FRONTEND=noninteractive
# Wait for network briefly before apt (fresh LXC may need a few seconds after pct start)
for i in $(seq 1 12); do
  if getent hosts pypi.org >/dev/null 2>&1 || getent hosts wheels.vllm.ai >/dev/null 2>&1 || ping -c1 -W2 1.1.1.1 >/dev/null 2>&1; then
    break
  fi
  echo "  Waiting for network... ($i/12)"
  sleep 3
done
apt-get update
apt-get install -y --no-install-recommends curl ca-certificates git openssh-server \
  libgomp1 libnuma1 libatomic1 libdrm2 python3-dev build-essential \
  libopenmpi3 openmpi-bin libopenmpi-dev gnupg 2>&1 || true
# OpenMPI provides libmpi.so.40 / libmpi_cxx.so.40 required by torch's bundled deps
# gnupg is REQUIRED to dearmor the ROCm repo key — without it a 0-byte keyring is written
# and apt silently rejects the repo ("NO_PUBKEY"), so no ROCm libs install and torch import fails.

# Resolve versions AFTER network + curl are ready (single source of truth — deploy does same, but we trust pinned)
echo "[1/8] Resolving vLLM + ROCm variant (pinned: VLLM_VERSION=${VLLM_VERSION:-auto} VLLM_ROCM_VARIANT=${VLLM_ROCM_VARIANT:-auto})..."
RESOLVED_VLLM_VERSION="$(resolve_vllm_version "${VLLM_VERSION}")"
RESOLVED_VARIANT="$(resolve_rocm_variant "${RESOLVED_VLLM_VERSION}" "${VLLM_ROCM_VARIANT}")"
RESOLVED_PYTHON="$(resolve_python_version "${RESOLVED_VLLM_VERSION}" "${RESOLVED_VARIANT}")"
# Honor explicit PYTHON_VERSION if pinned and wheels python differs, warn
if [[ "${PYTHON_VERSION}" != "${RESOLVED_PYTHON}" && -n "${PYTHON_VERSION:-}" ]]; then
  echo "WARNING: PYTHON_VERSION=${PYTHON_VERSION} pinned but wheels want ${RESOLVED_PYTHON} — using pinned ${PYTHON_VERSION}" >&2
  RESOLVED_PYTHON="${PYTHON_VERSION}"
fi
if [[ -z "${PYTHON_VERSION:-}" ]]; then
  PYTHON_VERSION="${RESOLVED_PYTHON}"
else
  PYTHON_VERSION="${RESOLVED_PYTHON}"
fi

echo "  Resolved VLLM          : ${RESOLVED_VLLM_VERSION}"
echo "  Resolved ROCm variant  : ${RESOLVED_VARIANT} ($(variant_to_dotted "${RESOLVED_VARIANT}"))"
echo "  Resolved Python        : ${PYTHON_VERSION}"
echo "  Wheels index           : ${WHEELS_BASE}/${RESOLVED_VLLM_VERSION}/${RESOLVED_VARIANT}/"

VLLM_INDEX="${WHEELS_BASE}/${RESOLVED_VLLM_VERSION}/${RESOLVED_VARIANT}/"

# ROCm runtime libs (needed by torch wheels — they link to host libs like MIOpen/rocBLAS)
_ROCM_DOTTED="$(variant_to_dotted "${RESOLVED_VARIANT}")"
_ROCM_MAJOR="$(echo "${_ROCM_DOTTED}" | cut -d. -f1)"
echo "[1/8] Installing ROCm runtime libs for ${_ROCM_DOTTED} (if needed)..."
# gpg (gnupg) is REQUIRED to dearmor the repo key — hard gate. (Bug 0.5.0: missing gpg wrote a
# 0-byte keyring, apt silently rejected the repo, 'Unable to locate package', torch import died.)
command -v gpg >/dev/null 2>&1 || { echo "FATAL: gpg (gnupg) not installed — cannot import ROCm repo key." >&2; exit 1; }
# Always (re)write repo — previous runs may have wrong suite (ubuntu vs noble) and cause 404
mkdir -p /etc/apt/keyrings
# Fetch + dearmor ROCm GPG key with verification — fail loudly on empty/invalid keyring
fetch_rocm_key() {
  local url tmp="/tmp/amdrocm-key.asc"
  for url in "$@"; do
    echo "  Trying key: ${url}"
    if curl -fsSL --connect-timeout 10 --max-time 60 --retry 3 --retry-delay 3 "$url" -o "$tmp" 2>/dev/null && [[ -s "$tmp" ]]; then
      if gpg --batch --yes --dearmor -o /etc/apt/keyrings/amdrocm.gpg "$tmp" 2>/dev/null && [[ -s /etc/apt/keyrings/amdrocm.gpg ]]; then
        gpg --batch --show-keys /etc/apt/keyrings/amdrocm.gpg >/dev/null 2>&1 && { echo "  Key imported ($(stat -c%s /etc/apt/keyrings/amdrocm.gpg) bytes)"; return 0; }
      fi
    fi
  done
  return 1
}
_ROCM_REPO_URL="" _ROCM_SUITE=""
if [[ "${_ROCM_MAJOR}" -ge 10 ]] 2>/dev/null; then
  echo "  Adding ROCm apt repo for ${_ROCM_DOTTED} (stable.repo.amd.com) ..."
  fetch_rocm_key "https://stable.repo.amd.com/rocm/gpg/packages.gpg" \
                 "https://repo.radeon.com/rocm/rocm.gpg.key" || \
    { echo "FATAL: could not fetch/validate ROCm GPG key — check network to stable.repo.amd.com" >&2; exit 1; }
  # 10.x repo uses ubuntuNNNN dists (not codenames), suite 'stable' — probe before writing
  _VER_ID="$(. /etc/os-release 2>/dev/null; echo "${VERSION_ID:-24.04}")"
  for _dist in "ubuntu${_VER_ID//./}" "noble"; do
    if curl -fsSL -o /dev/null --connect-timeout 10 --max-time 30 "https://stable.repo.amd.com/rocm/core/packages/${_dist}/dists/stable/Release" 2>/dev/null; then
      _ROCM_REPO_URL="https://stable.repo.amd.com/rocm/core/packages/${_dist}"; _ROCM_SUITE="stable"; break
    fi
  done
else
  echo "  Adding ROCm apt repo for ${_ROCM_DOTTED} (repo.radeon.com) ..."
  fetch_rocm_key "https://repo.radeon.com/rocm/rocm.gpg.key" \
                 "https://repo.amd.com/rocm/rocm.gpg.key" || \
    { echo "FATAL: could not fetch/validate ROCm GPG key — check network to repo.radeon.com" >&2; exit 1; }
  _CODENAME="$(. /etc/os-release 2>/dev/null; echo "${VERSION_CODENAME:-noble}")"
  [[ -z "${_CODENAME}" ]] && _CODENAME="noble"
  # Probe full dotted (7.2.3) first for exact wheel match, then major.minor (7.2)
  for _ver in "${_ROCM_DOTTED}" "${_ROCM_DOTTED%.*}"; do
    if curl -fsSL -o /dev/null --connect-timeout 10 --max-time 30 "https://repo.radeon.com/rocm/apt/${_ver}/dists/${_CODENAME}/Release" 2>/dev/null; then
      _ROCM_REPO_URL="https://repo.radeon.com/rocm/apt/${_ver}"; _ROCM_SUITE="${_CODENAME}"; break
    fi
  done
fi
if [[ -z "${_ROCM_REPO_URL}" ]]; then
  echo "FATAL: no reachable ROCm apt repo for ${_ROCM_DOTTED} (probed stable.repo.amd.com + repo.radeon.com)" >&2
  exit 1
fi
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] ${_ROCM_REPO_URL} ${_ROCM_SUITE} main" > /etc/apt/sources.list.d/rocm.list
echo 'APT::Key::GPGCommand "/usr/bin/gpg";' > /etc/apt/apt.conf.d/99gpg-override
echo "  Repo: $(cat /etc/apt/sources.list.d/rocm.list)"
set +e
apt-get update -o Acquire::Check-Valid-Until=false 2>&1 | tee /tmp/apt-update-rocm.log | tail -15
_auc=${PIPESTATUS[0]}
set -e
if ! apt-cache policy rocm-core 2>/dev/null | grep -qE 'Candidate: [0-9]'; then
  echo "FATAL: ROCm apt repo not usable (apt update rc=${_auc}) — no 'rocm-core' candidate." >&2
  echo "--- apt update log (tail) ---" >&2; tail -30 /tmp/apt-update-rocm.log >&2 || true
  echo "--- rocm.list ---" >&2; cat /etc/apt/sources.list.d/rocm.list >&2 || true
  echo "--- keyring ---" >&2; ls -la /etc/apt/keyrings/amdrocm.gpg >&2 || true
  exit 1
fi
echo "  rocm-core candidate: $(apt-cache policy rocm-core | awk '/Candidate/{print $2}')"
echo "  Installing ROCm libs for ${_ROCM_DOTTED} (covers libroctx/MIOpen/rocBLAS missing)..."
# Don't pipe to tail directly (masks exit code). Use tee + PIPESTATUS.
set +e
apt-get install -y --no-install-recommends rocm 2>&1 | tee /tmp/rocm-install.log; _rc=${PIPESTATUS[0]}
if [[ $_rc -ne 0 ]]; then
  echo "  'rocm' meta failed (rc=$_rc), trying minimal ROCm libs..."
  tail -20 /tmp/rocm-install.log || true
  apt-get install -y --no-install-recommends rocm-core rocm-hip-runtime rocblas hipblas miopen-hip rccl 2>&1 | tee /tmp/rocm-min.log; _rc2=${PIPESTATUS[0]}
  if [[ $_rc2 -ne 0 ]]; then
    echo "  minimal meta failed (rc=$_rc2), trying individual libs..."
    tail -20 /tmp/rocm-min.log || true
    # hsa-amd-aqlprofile (libhsa-amd-aqlprofile64.so.1) + libdw1 (libdw.so.1) are torch wheel deps
    apt-get install -y --no-install-recommends \
      hip-runtime-amd rocblas hipblas hipblaslt hipfft hiprand hipsolver hipsparse hipsparselt \
      rccl miopen-hip roctracer rocprofiler-sdk rocsolver hsa-amd-aqlprofile libdw1 2>&1 | tee /tmp/rocm-indiv.log; _rc3=${PIPESTATUS[0]}
    if [[ $_rc3 -ne 0 ]]; then
      echo "FATAL: ROCm runtime libs install failed (rc=$_rc3) — torch import would fail." >&2
      tail -30 /tmp/rocm-indiv.log >&2 || true
      echo "--- apt policy debug ---" >&2
      apt-cache policy rocm hip-runtime-amd 2>&1 | head -40 >&2 || true
      cat /etc/apt/sources.list.d/rocm.list >&2 || true
      exit 1
    fi
  fi
fi
set -e
# ROCm installs to /opt/rocm-<ver> (+ /opt/rocm symlink) — ensure ld.so sees it (derive, don't hardcode)
# NOTE: '|| true' inside the pipeline is mandatory — with set -o pipefail, an unmatched /opt/rocm*/lib64
# glob makes ls exit 2 and kills the whole script silently (2>/dev/null hides the reason).
{ ls -d /opt/rocm*/lib /opt/rocm*/lib64 2>/dev/null || true; } | sort -u > /etc/ld.so.conf.d/rocm.conf
[[ -s /etc/ld.so.conf.d/rocm.conf ]] || { echo "FATAL: no /opt/rocm*/lib dirs after ROCm install — where did apt put ROCm?" >&2; exit 1; }
ldconfig
# Derive loader path from what actually exists (no hardcoded version dirs)
ROCM_LIB_DIRS="$(paste -sd: /etc/ld.so.conf.d/rocm.conf)"
export LD_LIBRARY_PATH="${ROCM_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
_ROCM_HOME="$( { ls -d /opt/rocm-[0-9]* 2>/dev/null || true; } | sort -V | tail -1)"
[[ -z "${_ROCM_HOME}" ]] && _ROCM_HOME="/opt/rocm"
# Early gate: core libs must resolve NOW (fail in ~1 min, not after ~7 min of pip install)
# Cache ldconfig -p once (grep -q in a pipe under pipefail can false-fail via SIGPIPE)
_LDCACHE="$(ldconfig -p 2>/dev/null || true)"
for _lib in libamdhip64.so.7 libroctx64.so.4 librocblas.so.5 libMIOpen.so.1; do
  grep -q "${_lib}" <<<"${_LDCACHE}" || { echo "FATAL: ${_lib} not resolvable after ROCm install. rocm.conf: $(tr '\n' ' ' < /etc/ld.so.conf.d/rocm.conf)" >&2; exit 1; }
done
echo "  ROCm libs OK: $(grep -cE 'roctx|MIOpen|rocblas|hipblas|rccl' <<<"${_LDCACHE}" || true) shared objects resolvable (${ROCM_LIB_DIRS})"

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

# --- 3. INSTALL vLLM (simple one-liner, starting point) ---
echo "[3/8] Installing vLLM ${RESOLVED_VLLM_VERSION} ${RESOLVED_VARIANT} from ${VLLM_INDEX} ..."
echo "  Running: uv pip install vllm --extra-index-url ${VLLM_INDEX}"
# The ROCm index is self-contained (torch, triton, amdsmi, etc.) — one extra-index covers the whole stack.
# We use --extra-index-url so the +rocm local version wins over PyPI CUDA build (pip behavior, uv also honors local version).
uv pip install --python "${PY}" vllm --extra-index-url "${VLLM_INDEX}"
# Ensure Prometheus metrics endpoint is available at /metrics (vllm serve exposes it by default via prometheus_client - no --enable-metrics flag)
uv pip install --python "${PY}" prometheus_client

# Optional but recommended on gfx1150 APU: flash-attn / aiter for attention kernel (speed, not memory)
# Warn-and-continue if unavailable — Triton fallback via FLASH_ATTENTION_TRITON_AMD_ENABLE.
echo "[3/8] Installing flash-attn + amd-aiter (optional, warn-only)..."
uv pip install --python "${PY}" --extra-index-url "${VLLM_INDEX}" "flash-attn" "amd-aiter" 2>&1 | tail -20 || echo "WARNING: flash-attn/aiter not available from ${VLLM_INDEX}, will use Triton fallback"

# --- 4. HARD VERIFICATION (no CUDA) ---
echo "[4/8] Verifying vLLM install..."
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
# Ensure ROCm libs are on loader path for this check (and for vllm-run.sh later)
export LD_LIBRARY_PATH="${ROCM_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
ldconfig 2>&1 || true

PKGS="$("${PY}" -m pip list --format=freeze)"
if grep -qiE '^(nvidia-|cuda-)' <<<"${PKGS}"; then
  echo "FATAL: CUDA packages found in the venv — something pulled the CUDA build of vLLM/torch." >&2
  grep -iE '^(nvidia-|cuda-)' <<<"${PKGS}" >&2 || true
  exit 1
fi

MISSING="$(ldd "${VENV_DIR}"/lib/python*/site-packages/torch/lib/*.so 2>/dev/null | grep 'not found' | sort -u || true)"
if [[ -n "${MISSING}" ]]; then
  echo "FATAL: missing shared libs for torch (ROCm apt libs incomplete):" >&2
  echo "${MISSING}" >&2
  echo "  rocm.conf: $(tr '\n' ' ' < /etc/ld.so.conf.d/rocm.conf 2>/dev/null)" >&2
  exit 1
fi

"${PY}" - <<'EOF'
import sys, torch
print("torch        :", torch.__version__)
print("torch.hip    :", torch.version.hip)
if "rocm" not in torch.__version__ and not torch.version.hip:
    sys.exit("FATAL: torch is not a ROCm build")
if not torch.cuda.is_available():
    sys.exit("FATAL: GPU not visible to torch. Check LXC passthrough (/dev/kfd, /dev/dri), groups, and host amdgpu driver.")
print("device       :", torch.cuda.get_device_name(0))
free, total = torch.cuda.mem_get_info()
print(f"gpu memory   : {free/2**30:.1f} GiB free / {total/2**30:.1f} GiB total (iGPU = shared system RAM)")
import vllm
print("vllm         :", vllm.__version__)
try:
    import flash_attn
    print("flash-attn   :", flash_attn.__version__)
except Exception as e:
    print("flash-attn   : not installed (Triton fallback)", e)
EOF

# --- 5. OPEN-WEBUI VENV (native, same LXC, no docker) ---
echo "[5/8] Installing Open WebUI (native, port ${WEBUI_PORT})..."
rm -rf "${WEBUI_VENV_DIR}"
uv venv --seed --python "${PYTHON_VERSION}" "${WEBUI_VENV_DIR}"
WEBUI_PY="${WEBUI_VENV_DIR}/bin/python"
# open-webui pulls many deps — use PyPI (default) + same ROCm index extra if needed
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
# Prometheus metrics always exposed at http://${AI_PORT}/metrics (unauth on vmbr0, no flag needed - prometheus_client)
# Optional: require this bearer token on the API (recommended, host is 0.0.0.0)
AI_API_KEY=""
# Extra 'vllm serve' args, space-separated. Example: "--max-num-seqs 8 --skip-mm-profiling"
AI_EXTRA_ARGS=""
# Real vLLM variable:
VLLM_LOGGING_LEVEL=INFO
# gfx1150 override — only set if kernels fail to load with "invalid device function":
# HSA_OVERRIDE_GFX_VERSION=11.0.0
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
# vllm-run.sh: ExecStart for vllm.service (native ROCm wheels in ${VENV_DIR})
set -euo pipefail
set -a; . ${ENV_FILE}; set +a

export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export HF_HUB_CACHE="${MODEL_DIR}/.hf-cache"
export PATH="${VENV_DIR}/bin:\${PATH}"
export LD_LIBRARY_PATH="${ROCM_LIB_DIRS}:\${LD_LIBRARY_PATH:-}"
export ROCM_PATH="${_ROCM_HOME}"
export HIP_PATH="${_ROCM_HOME}"

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
Description=vLLM OpenAI-compatible server (native ROCm, no docker)
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

[Bootstrap complete: vLLM ${RESOLVED_VLLM_VERSION}/${RESOLVED_VARIANT} + Open WebUI native, script v0.5.0]
  vLLM API   : http://<container-ip>:${AI_PORT}/v1   (health: /health)
  Open WebUI : http://<container-ip>:${WEBUI_PORT}/  (chat UI, BYPASS_MODEL_ACCESS_CONTROL=true)
  Model      : ${DEFAULT_MODEL_PATH} (served as ${DEFAULT_MODEL_NAME})
  Config     : ${ENV_FILE}  +  ${WEBUI_ENV_FILE}
  Switch     : ${SWITCH_SCRIPT}
  GPU        : gfx1150, ROCm ${_ROCM_DOTTED} libs at ${ROCM_LIB_DIRS} via AMD apt + wheels ${VLLM_INDEX}
  Logs       : journalctl -u vllm -f  |  journalctl -u open-webui -f
SUMMARY
