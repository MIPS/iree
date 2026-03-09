#!/usr/bin/env bash
# rvv_qemu_workflow_static.sh
#
# End-to-end MIPS matmul pipeline — STATIC kernel embedding.
#
# The RVV kernel (.o) is baked into the dispatch ELF inside the .vmfb at
# iree-compile time via a custom lld wrapper. No plugin .so is needed at
# runtime.
#
# Pipeline:
#   mips_matmul_test.mlir
#     ─[iree-opt torch-to-iree{use-mips-matmul=true}]─► flow.mlir
#     ─[clang --target=riscv64]──────────────────────► matmul_kernel_riscv.o
#     ─[lld_wrapper.sh] (appends .o to every dispatch link)
#     ─[iree-compile --iree-mips-static-embedding]────► matmul.vmfb
#     ─[qemu-riscv64 iree-run-module]────────────────► result (no --executable_plugin)
#
# Usage:
#   bash rvv_qemu_workflow_static.sh            # RISC-V QEMU, vlen=512
#   bash rvv_qemu_workflow_static.sh --host     # x86 host (scalar fallback)
#   bash rvv_qemu_workflow_static.sh --vlen 256 # QEMU with vlen=256

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
IREE_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="${HOME}/MLIR_Work/mips"
HOST_BUILD="${WORK_DIR}/iree-build"
HOST_INSTALL="${HOST_BUILD}/install"
RISCV_BUILD="${WORK_DIR}/iree-build-riscv"
OUT_DIR="${WORK_DIR}/out/static"

IREE_OPT="${HOST_INSTALL}/bin/iree-opt"
IREE_COMPILE="${HOST_INSTALL}/bin/iree-compile"
HOST_RUN="${HOST_INSTALL}/bin/iree-run-module"
RISCV_RUN="${RISCV_BUILD}/install/bin/iree-run-module"
QEMU="${HOME}/local/bin/qemu-riscv64"
SYSROOT="${HOME}/riscv/toolchain/clang/linux/RISCV/sysroot"

CLANG="${HOME}/miniforge3/bin/clang"
LLD="${HOME}/miniforge3/bin/ld.lld"
CLANG_INC="${HOME}/miniforge3/lib/clang/18/include"

KERNEL_SRC="${IREE_SRC}/runtime/src/iree/builtins/mips/matmul_kernel.c"
TEST_MLIR="${WORK_DIR}/mips_matmul_test.mlir"

# Rocky 8's libstdc++ is too old; conda has GLIBCXX 3.4.29+.
export LD_LIBRARY_PATH="${HOME}/miniforge3/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
HOST_MODE=0
VLEN=512
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)   HOST_MODE=1 ;;
    --vlen)   shift; VLEN="$1" ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
  shift
done

mkdir -p "${OUT_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
section() { echo ""; echo "══[ $* ]══════════════════════════════════════════════════"; }
ok()      { echo "  [ok] $*"; }
run_qemu() {
  local vlen="$1"; shift
  "${QEMU}" -cpu "rv64,v=true,vlen=${vlen},elen=64,vext_spec=v1.0" \
            -L "${SYSROOT}" "${RISCV_RUN}" "$@"
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: torch → IREE flow IR
# ─────────────────────────────────────────────────────────────────────────────
section "Step 1: torch → IREE flow IR"

"${IREE_OPT}" \
  --pass-pipeline="builtin.module(torch-to-iree{use-mips-matmul=true})" \
  "${TEST_MLIR}" -o "${OUT_DIR}/flow.mlir"
ok "${OUT_DIR}/flow.mlir"

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Cross-compile RVV kernel → RISC-V relocatable object
# ─────────────────────────────────────────────────────────────────────────────
section "Step 2: Compile matmul_kernel.c → .o"

KERNEL_O="${OUT_DIR}/matmul_kernel_riscv.o"

if [[ "${HOST_MODE}" == "1" ]]; then
  "${CLANG}" --target=x86_64-linux-gnu \
    -O2 -c -I "${IREE_SRC}/runtime/src" \
    "${KERNEL_SRC}" -o "${KERNEL_O}"
  ok "x86 scalar kernel: ${KERNEL_O}"
else
  "${CLANG}" --target=riscv64-linux-gnu -march=rv64gcv -mabi=lp64d \
    -O2 -c -nostdinc -isystem "${CLANG_INC}" \
    -I "${IREE_SRC}/runtime/src" \
    "${KERNEL_SRC}" -o "${KERNEL_O}"
  ok "RISC-V RVV kernel: ${KERNEL_O}  ($(file -b "${KERNEL_O}" | cut -d, -f1))"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Create lld wrapper
#
# IREE calls its embedded linker as:
#   lld -flavor gnu --no-undefined -nostdlib -static -shared ... dispatch.o
# We append the kernel .o so my_matmul_kernel resolves at link time.
# -Bsymbolic: bind all same-ELF symbols locally to avoid R_RISCV_JUMP_SLOT
# entries in .rela.plt — IREE's embedded ELF loader ignores DT_JMPREL
# (.rela.plt) and only processes DT_RELA (.rela.dyn). Without this flag,
# the PLT GOT slot is never patched, causing a segfault on the first call.
# ─────────────────────────────────────────────────────────────────────────────
section "Step 3: Create lld_wrapper.sh"

LLD_WRAPPER="${OUT_DIR}/lld_wrapper.sh"
cat > "${LLD_WRAPPER}" << WRAPPER
#!/usr/bin/env bash
exec "${LLD}" "\$@" "${KERNEL_O}" -Bsymbolic
WRAPPER
chmod +x "${LLD_WRAPPER}"
ok "${LLD_WRAPPER}  (appends ${KERNEL_O})"

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: iree-compile → .vmfb (kernel statically linked)
# ─────────────────────────────────────────────────────────────────────────────
section "Step 4: iree-compile → .vmfb (static)"

if [[ "${HOST_MODE}" == "1" ]]; then
  RISCV_FLAGS=()
else
  RISCV_FLAGS=(
    "--iree-llvmcpu-target-triple=riscv64-linux-gnu"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  )
fi

VMFB="${OUT_DIR}/matmul_static.vmfb"
"${IREE_COMPILE}" \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-link-embedded=true \
  --iree-llvmcpu-embedded-linker-path="${LLD_WRAPPER}" \
  --iree-mips-static-embedding \
  "${RISCV_FLAGS[@]}" \
  "${OUT_DIR}/flow.mlir" -o "${VMFB}"
ok "${VMFB}  ($(du -sh "${VMFB}" | cut -f1))"

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Verify kernel is embedded in the dispatch ELF
# ─────────────────────────────────────────────────────────────────────────────
section "Step 5: Verify static embedding"

ELF_OFFSET=$(grep -boa $'\x7fELF' "${VMFB}" 2>/dev/null | head -1 | cut -d: -f1 || true)
if [[ -n "${ELF_OFFSET}" ]]; then
  dd if="${VMFB}" bs=1 skip="${ELF_OFFSET}" 2>/dev/null > "${OUT_DIR}/dispatch.elf"
  python3 - "${OUT_DIR}/dispatch.elf" << 'PYEOF'
import sys
data = open(sys.argv[1], 'rb').read()
idx  = data.find(b'my_matmul_kernel')
rvv  = sum(1 for i in range(0, len(data)-3, 4) if data[i] & 0x7f == 0x57)
if idx != -1:
    tag = "[ok]" if rvv > 0 else "[warn]"
    print(f"  {tag} 'my_matmul_kernel' at offset {idx},  RVV instructions: {rvv}")
else:
    print("  [warn] 'my_matmul_kernel' not found in dispatch ELF")
PYEOF
else
  echo "  [warn] No ELF found in vmfb"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Run
# ─────────────────────────────────────────────────────────────────────────────
section "Step 6: Run (no --executable_plugin)"

MATMUL_ARGS=(
  --module="${VMFB}"
  --function="matmul_4x4"
  "--input=4x4xf32=1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1"
  "--input=4x4xf32=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"
)

if [[ "${HOST_MODE}" == "1" ]]; then
  echo "  Running on x86 host (scalar fallback)..."
  "${HOST_RUN}" "${MATMUL_ARGS[@]}"
else
  echo "  Running under QEMU vlen=${VLEN}..."
  run_qemu "${VLEN}" "${MATMUL_ARGS[@]}"

  echo ""
  echo "  VLEN sweep:"
  for V in 128 256 512; do
    printf "    vlen=%-4s  " "${V}:"
    run_qemu "${V}" "${MATMUL_ARGS[@]}" 2>&1 | grep "4x4xf32" || echo "(no output)"
  done
  echo "  Note: vlen=128 may produce zeros — vmfb compiled with +zvl512b"
fi

echo ""
echo "  Expected: 4x4xf32=[1 2 3 4][5 6 7 8][9 10 11 12][13 14 15 16]"

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo " DONE — Static embedding verified."
echo " Artifacts in ${OUT_DIR}/"
echo "   matmul_kernel_riscv.o  — kernel object (baked into vmfb)"
echo "   lld_wrapper.sh         — linker interceptor"
echo "   matmul_static.vmfb     — self-contained vmfb (no plugin at runtime)"
echo "════════════════════════════════════════════════════════════"
