# This script now installs OS-level build dependencies first (headers/tools
# needed by native-compiled Python packages like Triton), then creates a
# virtualenv and installs Python deps.
#
# OS prereqs (Debian/Ubuntu):
#   - python3-dev + python3.<X>-dev  (provides Python.h)
#   - build-essential                (compiler toolchain)
#   - ninja-build, pkg-config, libssl-dev
# ---------------------------------------------

# --- 0) Install OS prerequisites (Debian/Ubuntu) ---
echo "[bootstrap] Checking for apt-get to install OS prerequisites..."
if command -v apt-get >/dev/null 2>&1; then
  PY_MAJMIN=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
  echo "[bootstrap] Detected Python ${PY_MAJMIN}; will install python${PY_MAJMIN}-dev"

  echo "[bootstrap] Updating APT package lists..."
  sudo apt-get update -y

  echo "[bootstrap] Installing OS build dependencies..."
  sudo apt-get install -y \
    python3-dev \
    "python${PY_MAJMIN}-dev" \
    build-essential \
    ninja-build \
    pkg-config \
    libssl-dev
else
  echo "[bootstrap] apt-get not found — skipping OS prereqs install (non-Debian system)."
  echo "            If you are on another distro, install Python headers and build tools manually."
fi

# --- 1) Python virtual environment ---
VENV_DIR=${VENV_DIR:-".venv"}
if [[ -d "$VENV_DIR" ]]; then
  echo "[bootstrap] Reusing existing virtual environment at $VENV_DIR"
else
  echo "[bootstrap] Creating virtual environment in $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "[bootstrap] Upgrading pip + wheel..."
python -m pip install --upgrade pip wheel

# --- 2) Python dependencies ---
echo "[bootstrap] Installing core dependencies..."
pip install \
  "numpy==1.26.4" \
  pandas \
  networkx \
  scikit-learn \
  matplotlib \
  plotly \
  pydantic \
  fastapi \
  "uvicorn[standard]" \
  typer \
  rich \
  python-dotenv \
  pyyaml \
  sqlalchemy \
  chromadb \
  sentence-transformers \
  httpx \
  pytest

echo "[bootstrap] Installing PyTorch (CUDA 12.1)..."
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121

echo "[bootstrap] Installing PyG (CUDA 12.1)..."
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
  -f https://data.pyg.org/whl/torch-2.3.1+cu121.html

# --- 3) Freeze environment ---
echo "[bootstrap] Freezing requirements to requirements.txt..."
pip freeze > requirements.txt

# --- 4) Sanity checks ---
python - <<'PYEOF'
import sys
print("\n=== PYTHON ENV CHECK ===")
print(f"Python: {sys.version}")
try:
    import torch
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print("Torch import failed:", e)

try:
    import torch_geometric  # noqa: F401
    print("PyG import: OK")
except Exception as e:
    print("PyG import FAILED:", e)
    sys.exit(1)

print("\n[bootstrap] ✅ Environment looks good!")
PYEOF

echo "[bootstrap] Done. To activate this environment in new shells:"
echo "source $VENV_DIR/bin/activate"
