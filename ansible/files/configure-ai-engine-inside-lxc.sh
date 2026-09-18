#!/usr/bin/env bash
# configure-ai-engine-inside-lxc.sh (vLLM native variant)
# Version: 0.3.2
# Description: Bootstrap native vLLM (no docker) on Ubuntu 24.04 LXC
#              with ROCm userspace installed in-container (gfx1150).
# Target GPU: AMD Radeon 890M (gfx1150/Strix Halo) on Proxmox 9.x privileged LXC
# Requirements: Run as root inside privileged LXC with GPU passthrough
#               (/dev/dri/card0, renderD128, /dev/kfd) and /srv/ai/models bind mount
# Changelog:
#   0.3.2 - FIX: amdsmi pip vs lib mismatch (pip 7.0.2 wants amdsmi_set_gpu_clk_range
#           missing in libamd_smi.so 27.0.0 from ROCm 10.0; Grok correctly diagnosed
#           pip vs system ABI). Now install amdsmi from /opt/rocm/share/amd_smi
#           (matching system lib) or uninstall pip copy and rely on HIP fallback.
#   0.3.1 - FIX: VLLM_USE_V2_MODEL_RUNNER=0 (checkpoint.md proven: CUDA-wheel has no UVA
#           op get_cuda_view_from_cpu_tensor, V2 crashes; V1 runner works). Fix ROCm
#           paths (/opt/rocm vs /opt/rocm/core-10.0), pin triton to 3.4.0 to match
#           pytorch-triton-rocm, pip amdsmi for clean platform detection.
#   0.3.0 - PHASE 2 refactor: REMOVED DOCKER. vLLM now runs natively in LXC.
#           Installs ROCm userspace in-container (matching host ROCM_VERSION),
#           Python venv, vLLM via pip with ROCm support. Runtime config in /etc/vllm.env,
#           service launched by /usr/local/bin/vllm-run.sh under vllm.service.
#           Default model /srv/ai/models/Qwen3.5-9B (qwen3.5-9b, 18GB bf16 safetensors).
#           HSA_OVERRIDE_GFX_VERSION=11.0.0 (proven override for gfx1150).
#   0.2.0 - PHASE 1 refactor: vLLM runs as docker image vllm/vllm-openai-rocm.
#   0.1.x - Earlier venv-era with in-LXC ROCm debugging (see checkpoint.md).

set -euo pipefail

# --- CLEANUP: Stop any existing vllm service to avoid stale ExecStart ---
systemctl stop vllm 2>/dev/null || true
systemctl disable vllm 2>/dev/null || true
systemctl daemon-reload 2>/dev/null || true

# --- CONFIGURABLE (env-overridable, pushed via pct exec env ...) ---
ROCM_VERSION="${ROCM_VERSION:-10.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL_DIR="${MODEL_DIR:-/srv/ai/models}"
DEFAULT_MODEL_PATH="${DEFAULT_MODEL_PATH:-${MODEL_DIR}/Qwen3.5-9B}"
DEFAULT_MODEL_NAME="${DEFAULT_MODEL_NAME:-qwen3.5-9b}"
GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.40}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_SERVICE="/etc/systemd/system/vllm.service"
RUNNER="/usr/local/bin/vllm-run.sh"
VLLM_ENV="/etc/vllm.env"
SWITCH_SCRIPT="/usr/local/bin/vllm-switch-model.sh"
VENV_DIR="/opt/vllm-venv"

# Map ROCm version to repo dist (same logic as deploy script)
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
  if grep -qi "bullseye" /etc/os-release 2>/dev/null; then HOST_REPO_DIST="debian11"; fi
fi
[[ -z "${HOST_REPO_DIST}" ]] && HOST_REPO_DIST="debian13"

ROCM_MAJOR="$(echo "${ROCM_VERSION}" | cut -d. -f1)"
ROCM_MM="$(echo "${ROCM_VERSION}" | cut -d. -f1,2)"

# --- 1. BASE DEPENDENCIES ---
echo "[1/8] Installing base dependencies..."
apt-get update
apt-get install -y --no-install-recommends \
  curl ca-certificates gnupg software-properties-common \
  python3 python3-venv python3-pip \
  openssh-server

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

# --- 2. INSTALL ROCm USERSPACE IN LXC ---
echo "[2/8] Installing ROCm ${ROCM_VERSION} userspace in LXC (repo: ${HOST_REPO_DIST})..."
mkdir -p /etc/apt/keyrings
if [[ "${ROCM_MAJOR}" -ge 10 ]] 2>/dev/null; then
  echo "  Using stable.repo.amd.com for ROCm ${ROCM_VERSION} (${HOST_REPO_DIST})"
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
  echo "  Using repo.amd.com/packages-multi-arch for ROCm ${ROCM_VERSION} (${HOST_REPO_DIST})"
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

apt-get update -o Acquire::Check-Valid-Until=false 2>&1 | tee /tmp/rocm-apt-update.log || true

# Install ROCm userspace packages (minimal set for vLLM)
echo "  Installing ROCm userspace packages..."
# Package names don't have version suffix in name; version is in package version
apt-get install -y --no-install-recommends \
  amdrocm amdrocm-core \
  hipcc libhiprtc-builtins5 \
  amdrocm-llvm \
  rocminfo \
  rocm-smi \
  amdrocm-amdsmi 2>&1 || true

# Add ld.so.conf for ROCm libs (use /opt/rocm symlink which points to actual versioned dir)
echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf
# Also add versioned path if it exists (covers /opt/rocm/core-10.0 case)
if [[ -d "/opt/rocm-${ROCM_MM}" ]]; then
  echo "/opt/rocm-${ROCM_MM}/lib" >> /etc/ld.so.conf.d/rocm.conf
fi
if [[ -d "/opt/rocm/core-${ROCM_MM}" ]]; then
  echo "/opt/rocm/core-${ROCM_MM}/lib" >> /etc/ld.so.conf.d/rocm.conf
fi
ldconfig

# Verify ROCm in LXC
echo "[2/8] Verifying ROCm in LXC..."
rocm-smi 2>&1 | head -20 || true
rocminfo 2>&1 | head -30 || true

# --- 3. PYTHON VENV + vLLM ---
echo "[3/8] Creating Python venv at ${VENV_DIR}..."
python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip setuptools wheel

# Install torch with ROCm support first
echo "[3/8] Installing PyTorch with ROCm ${ROCM_MM}..."
# Pin to known ROCm-compatible version (from checkpoint.md: torch 2.8.0+rocm6.4 works)
# vLLM 0.29.0 may require newer torch, but ROCm builds lag. Use 2.8.0+rocm6.4 for ROCm 10.x
if [[ "${ROCM_MAJOR}" -ge 10 ]] 2>/dev/null; then
  "${VENV_DIR}/bin/pip" install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/rocm6.4 \
    torch==2.8.0+rocm6.4 torchvision==0.23.0+rocm6.4 torchaudio==2.8.0+rocm6.4 2>&1 | tail -10
else
  # ROCm 7.x
  "${VENV_DIR}/bin/pip" install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/rocm5.7 \
    torch torchvision torchaudio 2>&1 | tail -10
fi

# Install vLLM with ROCm support (may overwrite torch — force ROCm torch back after)
echo "[3/8] Installing vLLM with ROCm support..."
"${VENV_DIR}/bin/pip" install --no-cache-dir \
  vllm[rocm] 2>&1 | tail -10
# vLLM 0.29 pulls CUDA torch 2.13 — force ROCm torch back
if [[ "${ROCM_MAJOR}" -ge 10 ]] 2>/dev/null; then
  echo "[3/8] Re-forcing ROCm torch 2.8.0+rocm6.4 (vLLM overwrites with CUDA)..."
  "${VENV_DIR}/bin/pip" install --no-cache-dir --force-reinstall \
    --index-url https://download.pytorch.org/whl/rocm6.4 \
    torch==2.8.0+rocm6.4 torchvision==0.23.0+rocm6.4 torchaudio==2.8.0+rocm6.4 2>&1 | tail -10
fi

# Pin triton to match pytorch-triton-rocm (3.7.1 breaks vllm's constexpr_function import)
echo "[3/8] Pinning triton to 3.4.0 to match pytorch-triton-rocm..."
"${VENV_DIR}/bin/pip" install --no-cache-dir --force-reinstall triton==3.4.0 2>&1 | tail -10 || true

# Fix amdsmi: pip 7.0.2 vs system lib 27.0.0 mismatch (undefined amdsmi_set_gpu_clk_range)
# Grok diagnosis correct — pip wheel from PyPI mismatches ROCm 10 lib. Prefer
# the amdsmi that ships with the installed ROCm (guaranteed ABI match).
echo "[3/8] Fixing amdsmi to match system ROCm lib..."
"${VENV_DIR}/bin/pip" uninstall -y amdsmi 2>&1 | tail -5 || true
if [[ -d "/opt/rocm/share/amd_smi" ]]; then
  echo "  Installing amdsmi from /opt/rocm/share/amd_smi (matches libamd_smi.so)..."
  "${VENV_DIR}/bin/pip" install --no-cache-dir /opt/rocm/share/amd_smi 2>&1 | tail -10 || echo "WARNING: amdsmi from /opt/rocm/share/amd_smi failed"
else
  echo "  /opt/rocm/share/amd_smi not found — skipping pip amdsmi, will rely on HIP fallback patch (vllm/platforms/__init__.py:411)"
fi
# Verify amdsmi import (must not crash torch)
if ! "${VENV_DIR}/bin/python" -c "import amdsmi; print('amdsmi OK', amdsmi.__version__ if hasattr(amdsmi,'__version__') else '')" 2>&1 | tail -5; then
  echo "WARNING: amdsmi still broken — uninstalling to let torch fallback (ModuleNotFoundError path)"
  "${VENV_DIR}/bin/pip" uninstall -y amdsmi 2>&1 | tail -5 || true
fi
# Final check: torch import must succeed (this is what vllm_rocm_accel_shim.pth does at startup)
echo "  Verifying torch+amdsmi import..."
"${VENV_DIR}/bin/python" -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip, 'cuda_available', torch.cuda.is_available())" 2>&1 | tail -10 || echo "WARNING: torch cuda import check failed"

# Apply ROCm compatibility patches (same as checkpoint.md proven fixes)
echo "[3/8] Applying ROCm compatibility patches..."
SP="${VENV_DIR}/lib/python3.12/site-packages/vllm"

# Patch 1: torch.accelerator shim for ROCm
python3 << 'PYEOF'
shim_content = '''# vllm_rocm_accel_shim.py - backfill torch.accelerator on ROCm builds
import torch
if not hasattr(torch, 'accelerator'):
    class _AcceleratorShim:
        @staticmethod
        def empty_cache():
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        @staticmethod
        def memory_stats(device=None):
            if hasattr(torch.cuda, 'memory_stats'):
                return torch.cuda.memory_stats(device)
            return {}
        @staticmethod
        def memory_allocated(device=None):
            if hasattr(torch.cuda, 'memory_allocated'):
                return torch.cuda.memory_allocated(device)
            return 0
        @staticmethod
        def max_memory_allocated(device=None):
            if hasattr(torch.cuda, 'max_memory_allocated'):
                return torch.cuda.max_memory_allocated(device)
            return 0
        @staticmethod
        def memory_reserved(device=None):
            if hasattr(torch.cuda, 'memory_reserved'):
                return torch.cuda.memory_reserved(device)
            return 0
        @staticmethod
        def reset_peak_memory_stats(device=None):
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats(device)
        @staticmethod
        def get_memory_info(device=None):
            if hasattr(torch.cuda, 'mem_get_info'):
                return torch.cuda.mem_get_info(device)
            return (0, 0)
        @staticmethod
        def empty_host_cache():
            pass
    torch.accelerator = _AcceleratorShim()
else:
    _acc = torch.accelerator
    if not hasattr(_acc, 'empty_cache'):
        _acc.empty_cache = lambda: torch.cuda.empty_cache() if hasattr(torch.cuda, 'empty_cache') else None
    if not hasattr(_acc, 'memory_stats'):
        _acc.memory_stats = lambda device=None: torch.cuda.memory_stats(device) if hasattr(torch.cuda, 'memory_stats') else {}
    if not hasattr(_acc, 'memory_allocated'):
        _acc.memory_allocated = lambda device=None: torch.cuda.memory_allocated(device) if hasattr(torch.cuda, 'memory_allocated') else 0
    if not hasattr(_acc, 'max_memory_allocated'):
        _acc.max_memory_allocated = lambda device=None: torch.cuda.max_memory_allocated(device) if hasattr(torch.cuda, 'max_memory_allocated') else 0
    if not hasattr(_acc, 'memory_reserved'):
        _acc.memory_reserved = lambda device=None: torch.cuda.memory_reserved(device) if hasattr(torch.cuda, 'memory_reserved') else 0
    if not hasattr(_acc, 'reset_peak_memory_stats'):
        _acc.reset_peak_memory_stats = lambda device=None: torch.cuda.reset_peak_memory_stats(device) if hasattr(torch.cuda, 'reset_peak_memory_stats') else None
    if not hasattr(_acc, 'get_memory_info'):
        _acc.get_memory_info = lambda device=None: torch.cuda.mem_get_info(device) if hasattr(torch.cuda, 'mem_get_info') else (0, 0)
    if not hasattr(_acc, 'empty_host_cache'):
        _acc.empty_host_cache = lambda: None
'''
with open("/opt/vllm-venv/lib/python3.12/site-packages/vllm_rocm_accel_shim.py", "w") as f:
    f.write(shim_content)
print("Wrote vllm_rocm_accel_shim.py")
PYEOF

# Create .pth to auto-load shim
echo "import vllm_rocm_accel_shim" > "${VENV_DIR}/lib/python3.12/site-packages/vllm_rocm_accel_shim.pth"

# Patch 2: SiluAndMul native fallback (two locations)
ACTIVATION_PY="${SP}/model_executor/layers/activation.py"
if [[ -f "${ACTIVATION_PY}" ]]; then
  python3 << 'PYEOF'
import re, pathlib
path = pathlib.Path("/opt/vllm-venv/lib/python3.12/site-packages/vllm/model_executor/layers/activation.py")
t = path.read_text()
# Patch 1: SiluAndMul (forward_native)
old1 = """        if (
            current_platform.is_cuda_alike()
            or current_platform.is_cpu()
            or current_platform.is_xpu()
        ):
            self.op = torch.ops._C.silu_and_mul"""
new1 = """        if (
            current_platform.is_cuda_alike()
            or current_platform.is_cpu()
            or current_platform.is_xpu()
        ):
            try:
                self.op = torch.ops._C.silu_and_mul
            except (AttributeError, RuntimeError):
                self._forward_method = self.forward_native"""
if old1 in t:
    t = t.replace(old1, new1)
    print("Patched SiluAndMul")
# Patch 2: SiluAndMulWithClamp / second occurrence (forward_native_with_clamp)
old2 = """        elif current_platform.is_cuda_alike():
            self.op = torch.ops._C.silu_and_mul"""
new2 = """        elif current_platform.is_cuda_alike():
            try:
                self.op = torch.ops._C.silu_and_mul
            except (AttributeError, RuntimeError):
                self._forward_method = self.forward_native_with_clamp"""
if old2 in t and "forward_native_with_clamp" not in t.split(old2)[0][-200:]:
    # only if not yet patched with correct fallback
    if old2 in t:
        t = t.replace(old2, new2)
        print("Patched second silu")
# fallback generic if patterns not matched but still bare assignment remains
if "self.op = torch.ops._C.silu_and_mul" in t and "try:" not in t[t.find("self.op = torch.ops._C.silu_and_mul")-100:t.find("self.op = torch.ops._C.silu_and_mul")+100]:
    print("Warning: unpatched silu_and_mul remains")
path.write_text(t)
print("activation.py patches done")
PYEOF
fi

# Patch 3: vllm_c provider gating
VLLM_C_PY="${SP}/kernels/vllm_c.py"
if [[ -f "${VLLM_C_PY}" ]]; then
  python3 << 'PYEOF'
path = "/opt/vllm-venv/lib/python3.12/site-packages/vllm/kernels/vllm_c.py"
with open(path) as f:
    content = f.read()
# Add C_EXT_AVAILABLE check and gate GPGPU_DEVICE
if "C_EXT_AVAILABLE = _c_ext_available()" not in content:
    # Find the GPGPU_DEVICE line and add check before it
    import re
    content = re.sub(
        r'(GPGPU_DEVICE\s*=\s*\(CUDA_ALIKE.*?\))',
        r'C_EXT_AVAILABLE = _c_ext_available()\n\1 and C_EXT_AVAILABLE',
        content
    )
    with open(path, 'w') as f:
        f.write(content)
    print("Patched vllm_c.py")
else:
    print("vllm_c.py already patched")
PYEOF
fi

# Patch 4: libtorch_cuda.so -> libtorch_hip.so symlink (ROCm torch lacks CUDA lib but vLLM tvm_ffi expects it)
echo "[3/8] Creating libtorch_cuda symlink for ROCm..."
if [[ -f "${VENV_DIR}/lib/python3.12/site-packages/torch/lib/libtorch_hip.so" ]]; then
  ln -sf "${VENV_DIR}/lib/python3.12/site-packages/torch/lib/libtorch_hip.so" "${VENV_DIR}/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so" 2>&1 || true
  echo "  Linked libtorch_hip.so -> libtorch_cuda.so"
fi

# Patch 5: torch_c_dlpack_ext fallback to cpu when cuda variant missing
python3 << 'PYEOF'
import pathlib
p = pathlib.Path('/opt/vllm-venv/lib/python3.12/site-packages/torch_c_dlpack_ext/core.py')
t = p.read_text()
old = '''    lib_path = (
        Path(__file__).parent
        / f"libtorch_c_dlpack_addon_torch{version.major}{version.minor}-{suffix}.{extension}"
    )
    if not lib_path.exists() or not lib_path.is_file():
        raise ImportError("No matching prebuilt torch c dlpack extension")
    lib = ctypes.CDLL(str(lib_path))'''
new = '''    lib_path = (
        Path(__file__).parent
        / f"libtorch_c_dlpack_addon_torch{version.major}{version.minor}-{suffix}.{extension}"
    )
    if not lib_path.exists() or not lib_path.is_file():
        suffix_fallback = "cpu"
        lib_path = Path(__file__).parent / f"libtorch_c_dlpack_addon_torch{version.major}{version.minor}-{suffix_fallback}.{extension}"
        if not lib_path.exists() or not lib_path.is_file():
            raise ImportError("No matching prebuilt torch c dlpack extension")
    try:
        lib = ctypes.CDLL(str(lib_path))
    except OSError as e:
        if suffix == "cuda":
            lib_path_cpu = Path(__file__).parent / f"libtorch_c_dlpack_addon_torch{version.major}{version.minor}-cpu.{extension}"
            lib = ctypes.CDLL(str(lib_path_cpu))
        else:
            raise'''
if old in t:
    t = t.replace(old, new)
    p.write_text(t)
    print('Patched torch_c_dlpack_ext/core.py')
else:
    print('torch_c_dlpack_ext already patched or pattern not found')
PYEOF

# Patch 6: tvm_ffi make torch_c_dlpack optional (ROCm: don't fail import if cuda lib missing)
python3 << 'PYEOF'
import pathlib
p = pathlib.Path('/opt/vllm-venv/lib/python3.12/site-packages/tvm_ffi/_optional_torch_c_dlpack.py')
t = p.read_text()
if 'except Exception as e:' not in t or '_LIB = None' not in t:
    old = '_LIB = load_torch_c_dlpack_extension()  # keep a reference to the loaded shared library'
    new = '''try:
        _LIB = load_torch_c_dlpack_extension()  # keep a reference to the loaded shared library
    except Exception as e:
        print(f"[warn] tvm_ffi torch_c_dlpack load failed (ROCm fallback): {e}")
        _LIB = None'''
    # handle indented version
    if '    _LIB = load_torch_c_dlpack_extension()' in t:
        old = '    _LIB = load_torch_c_dlpack_extension()  # keep a reference to the loaded shared library'
        new = '''    try:
        _LIB = load_torch_c_dlpack_extension()  # keep a reference to the loaded shared library
    except Exception as e:
        print(f"[warn] tvm_ffi torch_c_dlpack load failed (ROCm fallback): {e}")
        _LIB = None'''
    if old in t:
        t = t.replace(old, new)
        # fix double indent if needed
        t = t.replace('    try:\n        _LIB', '    try:\n        _LIB')
        p.write_text(t)
        print('Patched tvm_ffi/_optional_torch_c_dlpack.py')
    else:
        print('tvm_ffi pattern not found')
else:
    print('tvm_ffi already patched')
PYEOF
# fix indentation if botched
python3 << 'PYEOF'
import pathlib
p = pathlib.Path('/opt/vllm-venv/lib/python3.12/site-packages/tvm_ffi/_optional_torch_c_dlpack.py')
t = p.read_text()
old_bad = '''if os.environ.get("TVM_FFI_DISABLE_TORCH_C_DLPACK", "0") == "0":
    try:
    _LIB = load_torch_c_dlpack_extension()'''
new_good = '''if os.environ.get("TVM_FFI_DISABLE_TORCH_C_DLPACK", "0") == "0":
    try:
        _LIB = load_torch_c_dlpack_extension()'''
if old_bad in t:
    t = t.replace(old_bad, new_good)
    p.write_text(t)
    print('Fixed tvm_ffi indentation')
PYEOF

# Patch 7: vllm platform ROCm detection fallback via PyTorch HIP (when amdsmi pip not installed)
python3 << 'PYEOF'
import pathlib
p = pathlib.Path('/opt/vllm-venv/lib/python3.12/site-packages/vllm/platforms/__init__.py')
t = p.read_text()
old = """    except Exception as e:
        logger.debug("ROCm platform is not available because: %s", str(e))

    if not is_rocm and in_wsl():"""
new = """    except Exception as e:
        logger.debug("ROCm platform is not available because: %s", str(e))

    # Fallback: check PyTorch HIP directly (covers native ROCm without amdsmi pip)
    if not is_rocm:
        try:
            import torch
            if getattr(torch.version, "hip", None) and torch.cuda.is_available():
                is_rocm = True
                logger.debug("Confirmed ROCm platform is available via PyTorch HIP fallback.")
        except Exception as e2:
            logger.debug("ROCm HIP fallback detection failed because: %s", str(e2))

    if not is_rocm and in_wsl():"""
if old in t and "PyTorch HIP fallback" not in t:
    t = t.replace(old, new)
    p.write_text(t)
    print('Patched vllm/platforms/__init__.py for ROCm HIP fallback')
else:
    print('platform __init__.py already patched or pattern not found')
PYEOF

# --- 4. MODEL DIR ---
echo "[4/8] Checking model directory ${MODEL_DIR}..."
mkdir -p "${MODEL_DIR}"
mkdir -p "${MODEL_DIR}/.hf-cache" 2>&1 || true
if [ -d "${DEFAULT_MODEL_PATH}" ]; then
  echo "Default model present: ${DEFAULT_MODEL_PATH} ($(du -sh "${DEFAULT_MODEL_PATH}" 2>/dev/null | awk '{print $1}'))"
  ls -lh "${DEFAULT_MODEL_PATH}" 2>&1 | head -12 || true
else
  echo "WARNING: Default model ${DEFAULT_MODEL_PATH} not found — vLLM will fail to start until the model is present."
  echo "Available entries in ${MODEL_DIR}:"
  ls -lh "${MODEL_DIR}" 2>&1 | head -30 || true
fi

# GPU device nodes must be passed through
echo "[4/8] Pre-check: /dev/kfd + /dev/dri"
ls -l /dev/kfd /dev/dri/card0 /dev/dri/renderD128 2>&1 | head -10 || true
[[ -e /dev/kfd ]] || echo "WARNING: /dev/kfd missing — GPU passthrough not configured."

# --- 5. RUNTIME CONFIG (/etc/vllm.env) ---
echo "[5/8] Writing runtime config ${VLLM_ENV}..."
cat > "${VLLM_ENV}" << EOF
# vLLM runtime config (native, no docker). Edit, then: systemctl restart vllm
VLLM_PORT=${VLLM_PORT}
VLLM_MODEL_DIR=${MODEL_DIR}
VLLM_MODEL_PATH=${DEFAULT_MODEL_PATH}
VLLM_SERVED_NAME=${DEFAULT_MODEL_NAME}
VLLM_GPU_MEM_UTIL=${GPU_MEM_UTIL}
VLLM_MAX_MODEL_LEN=${MAX_MODEL_LEN}
HSA_OVERRIDE_GFX_VERSION=${GFX_VERSION}
VLLM_LOG_LEVEL=INFO
# VLLM_USE_V2_MODEL_RUNNER=0 is mandatory on ROCm with CUDA-wheel (no UVA op
# get_cuda_view_from_cpu_tensor) — see checkpoint.md §5.4. Do not set to 1.
VLLM_USE_V2_MODEL_RUNNER=0
# Extra vllm serve args (space-separated, no quoting). Examples:
#   VLLM_EXTRA_ARGS=--max-num-seqs 8 --skip-mm-profiling
VLLM_EXTRA_ARGS=
EOF
chmod 644 "${VLLM_ENV}"

# --- 6. NATIVE RUNNER + SYSTEMD UNIT ---
echo "[6/8] Writing ${RUNNER} and ${VLLM_SERVICE}..."
cat > "${RUNNER}" << 'RUNNER'
#!/usr/bin/env bash
# vllm-run.sh — ExecStart for vllm.service: launches native vLLM (no docker).
set -euo pipefail
set -a; . /etc/vllm.env; set +a

# Environment for ROCm - use /opt/rocm symlink (resolves to actual versioned dir)
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"
export VLLM_LOGGING_LEVEL="${VLLM_LOG_LEVEL}"
export HF_HUB_CACHE="/srv/ai/models/.hf-cache"
export PYTHONPATH="/opt/vllm-venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export PATH="/opt/vllm-venv/bin:/opt/rocm/bin:${PATH}"
export LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH:-}"
export ROCM_PATH="/opt/rocm"
export HIP_PATH="/opt/rocm"

# vLLM serve arguments
ARGS=(
  "${VLLM_MODEL_PATH}"
  --host 0.0.0.0
  --port "${VLLM_PORT}"
  --served-model-name "${VLLM_SERVED_NAME}"
  --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}"
  --max-model-len "${VLLM_MAX_MODEL_LEN}"
  --enforce-eager
  --trust-remote-code
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --limit-mm-per-prompt '{"image":1,"video":1}'
  --mm-processor-cache-gb 1
)
if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA <<< "${VLLM_EXTRA_ARGS}"
  ARGS+=("${EXTRA[@]}")
fi

exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
RUNNER
chmod 755 "${RUNNER}"

cat > "${VLLM_SERVICE}" << UNIT
[Unit]
Description=vLLM OpenAI-compatible server (native ROCm, no docker)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=-/etc/vllm.env
Environment=VLLM_USE_V2_MODEL_RUNNER=0
ExecStart=${RUNNER}
Restart=on-failure
RestartSec=10
User=root
# GPU devices passed through from host
# (Configured in /etc/pve/lxc/113.conf via deploy script)

[Install]
WantedBy=multi-user.target
UNIT

# --- 7. MODEL SWITCH HELPER ---
cat > "${SWITCH_SCRIPT}" << 'EOS'
#!/usr/bin/env bash
# vllm-switch-model.sh — switch the vLLM model (edits /etc/vllm.env, restarts service)
set -euo pipefail
ENV_FILE=/etc/vllm.env
MODEL_DIR=/srv/ai/models

echo "Current model: $(grep '^VLLM_MODEL_PATH=' "$ENV_FILE" | cut -d= -f2)"
echo ""
echo "Model dirs in ${MODEL_DIR}:"
ls -1 "${MODEL_DIR}" 2>/dev/null | grep -vE '^\.(hf-cache|cache)|^dl.*\.sh$|^switch|\.gguf$|\.bak' | sed 's/^/  /' || true
echo ""
read -rp "Model path (dir in ${MODEL_DIR}, or full path, or HF id): " NEW_MODEL
if [[ -z "${NEW_MODEL}" ]]; then echo "Aborted."; exit 1; fi
if [[ "${NEW_MODEL}" != /* && ! "${NEW_MODEL}" == */* && -d "${MODEL_DIR}/${NEW_MODEL}" ]]; then
  NEW_MODEL="${MODEL_DIR}/${NEW_MODEL}"
fi
if [[ -d "${NEW_MODEL}" ]]; then
  echo "Using local model dir: ${NEW_MODEL}"
else
  echo "Not a local dir — treating as HuggingFace id: ${NEW_MODEL}"
fi
read -rp "Served name [default: $(basename "${NEW_MODEL}" | tr '[:upper:]' '[:lower:]')]: " SERVED
SERVED="${SERVED:-$(basename "${NEW_MODEL}" | tr '[:upper:]' '[:lower:]')}"

sed -i -E "s|^VLLM_MODEL_PATH=.*|VLLM_MODEL_PATH=${NEW_MODEL}|" "$ENV_FILE"
sed -i -E "s|^VLLM_SERVED_NAME=.*|VLLM_SERVED_NAME=${SERVED}|" "$ENV_FILE"
echo "Updated:"
grep -E '^(VLLM_MODEL_PATH|VLLM_SERVED_NAME)=' "$ENV_FILE"

systemctl restart vllm
echo "Waiting for vLLM health on :8000 (model load from ZFS can take minutes)..."
for i in $(seq 1 90); do
  if curl -fsS -m 3 -o /dev/null http://127.0.0.1:8000/health 2>/dev/null; then
    echo "[OK] vLLM serving ${NEW_MODEL} as ${SERVED}"
    curl -s http://127.0.0.1:8000/v1/models | head -100 || true
    exit 0
  fi
  sleep 2
done
echo "WARNING: vLLM did not become healthy — check: journalctl -u vllm -f"
EOS
chmod +x "${SWITCH_SCRIPT}"
cp "${SWITCH_SCRIPT}" "${MODEL_DIR}/vllm-switch-model.sh" 2>&1 || true

# --- 8. ENABLE + START ---
echo "[8/8] Enabling and starting vllm service..."
systemctl daemon-reload
systemctl enable --now vllm 2>&1 || true

# --- 9. VERIFICATION ---
echo "[9/9] Verifying..."
echo "[vLLM health (waiting up to 5 min for model load) ...]"
HEALTHY=0
for i in $(seq 1 60); do
  if curl -fsS -m 3 -o /dev/null http://127.0.0.1:8000/health 2>/dev/null; then
    HEALTHY=1; break
  fi
  sleep 5
done
if [[ "${HEALTHY}" == "1" ]]; then
  echo "[health] OK"
  echo "[v1/models]"
  curl -s http://127.0.0.1:8000/v1/models | head -50 || true
else
  echo "[health] NOT healthy yet — vLLM may still be loading (18GB off ZFS) or hit a startup error."
  echo "  journalctl -u vllm -f        # service logs"
fi
echo ""
systemctl status vllm --no-pager 2>&1 | tail -15 || true
echo ""
echo "[Bootstrap complete - vllm 0.3.2 (native ROCm, no docker)]"
echo "  vLLM API (OpenAI) : http://<container-ip>:${VLLM_PORT}/v1  (health http://<container-ip>:${VLLM_PORT}/health)"
echo "  Model             : ${DEFAULT_MODEL_PATH} (served as ${DEFAULT_MODEL_NAME})"
echo "  Runtime config    : ${VLLM_ENV}  (edit + systemctl restart vllm)"
echo "  Switch model      : ${SWITCH_SCRIPT}"
echo "  GPU               : gfx1150 (890M) HSA_OVERRIDE_GFX_VERSION=${GFX_VERSION}"
echo "  ROCm in LXC       : ${ROCM_VERSION} (userspace), host provides amdgpu kernel driver"
echo "  Open WebUI        : NOT configured (phase 10)"
echo "  Logs              : journalctl -u vllm -f"