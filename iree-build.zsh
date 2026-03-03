#!/usr/bin/env zsh

# Exit if any command fails
set -e

# Paths (customize as needed)
SRC_DIR="$HOME/MLIR_Work/mips"
IREE_SRC_DIR="$SRC_DIR/iree"       # Path to your cloned iree
BUILD_DIR="$SRC_DIR/iree-build"

# Configuration
CMAKE_GENERATOR="Ninja"                # Change to "Unix Makefiles" if preferred
BUILD_TYPE="RelWithDebInfo"            # "Debug" or "RelWithDebInfo" are alternatives
NUM_JOBS=$(sysctl -n hw.logicalcpu)    # Uses all CPU cores

# Prepare directories
mkdir -p "${BUILD_DIR}"

# CMake configuration
cmake -S "${IREE_SRC_DIR}" -B "${BUILD_DIR}" \
    -G "${CMAKE_GENERATOR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
    -DIREE_ENABLE_SPLIT_DWARF=ON \
    -DIREE_ENABLE_LLD=ON \
    -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
    -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF \
    -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
    -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
    -DIREE_BUILD_PYTHON_BINDINGS=ON \
    -DPython3_EXECUTABLE="$(which python)"

# Build
ninja -C "${BUILD_DIR}" -j"${NUM_JOBS}"

# Or combine all steps using a utility target
#cmake --build ../iree-build --target iree-run-tests

echo "✅ IREE built and executed tests successfully!"
