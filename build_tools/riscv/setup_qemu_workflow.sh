#!/usr/bin/env bash
# setup_qemu_workflow.sh
#
# One-time setup for the MIPS/RVV QEMU workflow on a Linux host (Rocky 8).
# Installs toolchain, QEMU, and builds both IREE host and RISC-V targets.
#
# Steps (run all by default; pass --step=N to run one):
#   1. Install toolchain  — ninja, clang-18, lld-18 via conda-forge
#   2. Install sysroot    — RISC-V prebuilt sysroot + iree-run-module (RISC-V)
#   3. Build QEMU         — qemu-riscv64 user-mode from source
#   4. Build IREE (host)  — iree-opt, iree-compile, iree-run-module for x86
#   5. Build IREE (riscv) — iree-run-module cross-compiled for RISC-V
#
# Usage:
#   bash setup_qemu_workflow.sh           # run all steps
#   bash setup_qemu_workflow.sh --step=3  # run only QEMU build
#   bash setup_qemu_workflow.sh --step=4  # run only host IREE build

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit to match your environment
# ─────────────────────────────────────────────────────────────────────────────
WORK_DIR="${HOME}/MLIR_Work/mips"
IREE_SRC="${WORK_DIR}/iree"

HOST_BUILD="${WORK_DIR}/iree-build"          # iree-opt, iree-compile (x86)
HOST_INSTALL="${HOST_BUILD}/install"         # installed host tools
RISCV_BUILD="${WORK_DIR}/iree-build-riscv"   # iree-run-module (RISC-V)
QEMU_VER="8.2.2"
INSTALL_PREFIX="${HOME}/local"               # qemu-riscv64 installed here

CONDA="${HOME}/miniforge3/bin/conda"
CLANG="${HOME}/miniforge3/bin/clang"
CLANGXX="${HOME}/miniforge3/bin/clang++"
NINJA="${INSTALL_PREFIX}/bin/ninja"

# RISC-V prebuilt sysroot (downloaded by riscv_bootstrap.sh)
SYSROOT="${HOME}/riscv/toolchain/clang/linux/RISCV/sysroot"
RISCV_TOOLCHAIN="${HOME}/riscv/toolchain/clang/linux/RISCV"

# Rocky 8's system libstdc++ is too old; conda's copy has GLIBCXX 3.4.29+.
export LD_LIBRARY_PATH="${HOME}/miniforge3/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
STEP_ONLY=0
for arg in "$@"; do
  case "${arg}" in
    --step=*) STEP_ONLY="${arg#--step=}" ;;
    *) echo "Unknown arg: ${arg}"; exit 1 ;;
  esac
done

should_run() { [[ "${STEP_ONLY}" == "0" || "${STEP_ONLY}" == "$1" ]]; }

log()    { echo ""; echo "════════════════════════════════════════════════════════════"; echo " $*"; echo "════════════════════════════════════════════════════════════"; }
ok()     { echo "  [ok] $*"; }
skip()   { echo "  [skip] $*"; }
die()    { echo "  [FAIL] $*" >&2; exit 1; }

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Install toolchain (ninja, clang-18, lld-18 via conda-forge)
# ─────────────────────────────────────────────────────────────────────────────
step1_toolchain() {
  log "STEP 1: Install toolchain"

  # Miniforge (conda base)
  if [[ -x "${CONDA}" ]]; then
    skip "conda already at ${CONDA}"
  else
    local tmp; tmp="$(mktemp /tmp/miniforge_XXXXX.sh)"
    echo "  Downloading Miniforge..."
    curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
         -o "${tmp}"
    bash "${tmp}" -b -p "${HOME}/miniforge3"
    rm -f "${tmp}"
    ok "conda installed"
  fi

  # ninja
  if [[ -x "${NINJA}" ]]; then
    skip "ninja already at ${NINJA}"
  else
    local tmp; tmp="$(mktemp /tmp/ninja_XXXXX.zip)"
    curl -fsSL "https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-linux.zip" \
         -o "${tmp}"
    mkdir -p "${INSTALL_PREFIX}/bin"
    unzip -qo "${tmp}" -d "${INSTALL_PREFIX}/bin"
    chmod +x "${NINJA}"
    rm -f "${tmp}"
    ok "ninja installed"
  fi

  # clang-18 + lld-18
  if [[ -x "${CLANG}" ]]; then
    skip "clang already at ${CLANG} ($(${CLANG} --version | head -1))"
  else
    echo "  Installing clang-18 + lld-18 (this may take a few minutes)..."
    "${CONDA}" install -y -c conda-forge "clang=18" "clangxx=18" "lld=18" --no-update-deps
    ok "clang-18 + lld-18 installed"
  fi

  ok "Toolchain ready"
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Install RISC-V sysroot (IREE prebuilt)
# ─────────────────────────────────────────────────────────────────────────────
step2_sysroot() {
  log "STEP 2: Install RISC-V sysroot"

  if [[ -d "${SYSROOT}" ]]; then
    skip "Sysroot already at ${SYSROOT}"
    return
  fi

  echo "  Running riscv_bootstrap.sh (interactive — prompts for download paths)..."
  bash "$(dirname "${BASH_SOURCE[0]}")/riscv_bootstrap.sh"
  ok "Sysroot installed at ${SYSROOT}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Build QEMU riscv64-linux-user from source
# ─────────────────────────────────────────────────────────────────────────────
step3_qemu() {
  log "STEP 3: Build QEMU ${QEMU_VER} (riscv64-linux-user)"

  local qemu_bin="${INSTALL_PREFIX}/bin/qemu-riscv64"
  if [[ -x "${qemu_bin}" ]]; then
    skip "qemu-riscv64 already at ${qemu_bin} ($(${qemu_bin} --version | head -1))"
    return
  fi

  "${CONDA}" install -y -c conda-forge glib pkg-config 2>&1 | tail -3

  local tarball="${WORK_DIR}/qemu-${QEMU_VER}.tar.xz"
  if [[ ! -f "${tarball}" ]]; then
    echo "  Downloading QEMU ${QEMU_VER}..."
    curl -fsSL --progress-bar "https://download.qemu.org/qemu-${QEMU_VER}.tar.xz" -o "${tarball}"
  else
    skip "Tarball already downloaded"
  fi

  local src="${WORK_DIR}/qemu-${QEMU_VER}"
  if [[ ! -d "${src}" ]]; then
    echo "  Extracting QEMU source..."
    tar -xf "${tarball}" -C "${WORK_DIR}"
  fi

  export PKG_CONFIG_PATH="${HOME}/miniforge3/lib/pkgconfig:${HOME}/miniforge3/share/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
  export PKG_CONFIG="${HOME}/miniforge3/bin/pkg-config"
  export LDFLAGS="-Wl,-rpath,${HOME}/miniforge3/lib"

  echo "  Configuring QEMU..."
  cd "${src}"
  ./configure \
    --prefix="${INSTALL_PREFIX}" \
    --target-list="riscv64-linux-user" \
    --disable-system \
    --enable-linux-user \
    --disable-werror \
    --disable-docs \
    --disable-gtk \
    --disable-sdl \
    --disable-vnc \
    --disable-curl \
    --disable-capstone \
    --disable-kvm \
    --without-default-features \
    --enable-user

  echo "  Building QEMU ($(nproc) jobs)..."
  if [[ -f "${src}/build/build.ninja" ]]; then
    "${NINJA}" -C "${src}/build" -j"$(nproc)"
    "${NINJA}" -C "${src}/build" install
  else
    make -j"$(nproc)"
    make install
  fi

  ok "qemu-riscv64 installed at ${qemu_bin}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Build IREE host (iree-opt, iree-compile, iree-run-module for x86)
# ─────────────────────────────────────────────────────────────────────────────
step4_iree_host() {
  log "STEP 4: Build IREE (host — iree-opt, iree-compile, iree-run-module)"

  mkdir -p "${HOST_BUILD}"

  cmake -S "${IREE_SRC}" -B "${HOST_BUILD}" \
    -G Ninja \
    -DCMAKE_MAKE_PROGRAM="${NINJA}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER="${CLANG}" \
    -DCMAKE_CXX_COMPILER="${CLANGXX}" \
    -DCMAKE_ASM_COMPILER="${CLANG}" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_INSTALL_PREFIX="${HOST_INSTALL}" \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DIREE_ENABLE_SPLIT_DWARF=ON \
    -DIREE_ENABLE_LLD=ON \
    -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
    -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF \
    -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
    -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
    -DIREE_BUILD_PYTHON_BINDINGS=OFF \
    -DBENCHMARK_ENABLE_TESTING=OFF \
    -DHAVE_STD_REGEX=ON \
    -DHAVE_POSIX_REGEX=OFF

  echo "  Building ($(nproc) jobs)..."
  "${NINJA}" -C "${HOST_BUILD}" -j"$(nproc)" iree-opt iree-compile iree-run-module iree-tblgen

  echo "  Installing host tools to ${HOST_INSTALL}..."
  "${NINJA}" -C "${HOST_BUILD}" install/fast

  ok "iree-opt:           ${HOST_INSTALL}/bin/iree-opt"
  ok "iree-compile:       ${HOST_INSTALL}/bin/iree-compile"
  ok "iree-run-module:    ${HOST_INSTALL}/bin/iree-run-module"
  ok "iree-tblgen:        ${HOST_INSTALL}/bin/iree-tblgen"
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Build IREE RISC-V (iree-run-module cross-compiled for riscv64)
# ─────────────────────────────────────────────────────────────────────────────
step5_iree_riscv() {
  log "STEP 5: Build IREE (RISC-V cross — iree-run-module for riscv64)"

  [[ -f "${HOST_INSTALL}/bin/iree-tblgen" ]] || \
    die "Host install not found at ${HOST_INSTALL}/bin — run step 4 first."

  mkdir -p "${RISCV_BUILD}"

  cmake -S "${IREE_SRC}" -B "${RISCV_BUILD}" \
    -G Ninja \
    -DCMAKE_MAKE_PROGRAM="${NINJA}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_TOOLCHAIN_FILE="${IREE_SRC}/build_tools/cmake/riscv.toolchain.cmake" \
    -DIREE_HOST_BIN_DIR="${HOST_INSTALL}/bin" \
    -DRISCV_TOOLCHAIN_ROOT="${RISCV_TOOLCHAIN}" \
    -DIREE_BUILD_COMPILER=OFF \
    -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF \
    -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
    -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
    -DIREE_BUILD_PYTHON_BINDINGS=OFF \
    -DBENCHMARK_ENABLE_TESTING=OFF \
    -DCMAKE_INSTALL_PREFIX="${RISCV_BUILD}/install"

  echo "  Building ($(nproc) jobs)..."
  "${NINJA}" -C "${RISCV_BUILD}" -j"$(nproc)" iree-run-module
  "${NINJA}" -C "${RISCV_BUILD}" install/fast

  ok "iree-run-module (riscv64): ${RISCV_BUILD}/install/bin/iree-run-module"
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
should_run 1 && step1_toolchain
should_run 2 && step2_sysroot
should_run 3 && step3_qemu
should_run 4 && step4_iree_host
should_run 5 && step5_iree_riscv

echo ""
echo "════════════════════════════════════════════════════════════"
echo " Setup complete. Run the end-to-end workflows:"
echo ""
echo "   bash build_tools/riscv/rvv_qemu_workflow_static.sh"
echo "   bash build_tools/riscv/rvv_qemu_workflow_dynamic.sh"
echo "════════════════════════════════════════════════════════════"
