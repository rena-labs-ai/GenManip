#!/bin/bash
# Isaac Sim 6.0.1 + cuRobo v0.7.8 (sm_120) environment. No sudo, no system CUDA.
# Verified 2026-09-10 on RTX 5090 Laptop (sm_120), driver 580, Ubuntu 24.04 gcc 13.3.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d .venv ] || uv venv --python 3.12 .venv
PY=.venv/bin/python

uv pip install --python $PY torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
# --index-strategy/--prerelease are REQUIRED: plain resolution fails on mujoco-usd-converter / tinyobjloader.
uv pip install --python $PY "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match --prerelease=allow
# mplib skipped: it downgrades numpy to 1.26 (breaks Isaac Sim) and is rejected at runtime anyway.
grep -v '^mplib' requirements.txt | uv pip install --python $PY -r /dev/stdin

# --- cuRobo v0.7.8 (v1 API; GenManip imports curobo.wrap/curobo.types) built for sm_120 ---
# The pip wheels nvidia-cuda-nvcc-cu12 ship only ptxas/nvvm, no `nvcc` binary, so the toolchain
# comes from conda packages via a user-space micromamba (no sudo). Layout: .venv/cuda-12.8/.
mkdir -p saved/envs/bin
[ -x saved/envs/bin/micromamba ] || curl -sL https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xj -C saved/envs bin/micromamba
[ -x .venv/cuda-12.8/bin/nvcc ] || saved/envs/bin/micromamba create -y -p .venv/cuda-12.8 \
  -c nvidia -c conda-forge --override-channels "cuda-nvcc-impl=12.8" "cuda-cudart-dev=12.8" "cuda-cccl=12.8"
ln -sfn targets/x86_64-linux/include .venv/cuda-12.8/include   # torch cpp_extension expects $CUDA_HOME/include
ln -sfn lib .venv/cuda-12.8/lib64                              # ... and $CUDA_HOME/lib64
.venv/cuda-12.8/bin/nvcc --version | grep -q "release 12.8"

[ -d saved/envs/curobo ] || git clone --depth 1 --branch v0.7.8 https://github.com/NVlabs/curobo.git saved/envs/curobo
uv pip install --python $PY ninja
export CUDA_HOME=$PWD/.venv/cuda-12.8 PATH=$PWD/.venv/cuda-12.8/bin:$PATH TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=8
uv pip install --python $PY -e saved/envs/curobo --no-build-isolation   # ~10 min

# --- assets (accept the Omniverse EULA for every isaacsim run: export OMNI_KIT_ACCEPT_EULA=YES) ---
export OMNI_KIT_ACCEPT_EULA=YES
for d in basic banana banana-layout franka-robotiq; do $PY standalone_tools/download_assets.py --dataset $d; done
