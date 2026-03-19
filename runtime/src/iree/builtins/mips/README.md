# MIPS Kernel Library

This directory contains the hand-tuned kernel implementations for the
`mips` IREE dialect — a semantic dispatch layer that maps high-level tensor
operations to target-specific, optimized C kernels (currently RVV-vectorized
RISC-V matmul).

## Source Files

| File | Purpose |
|------|---------|
| `matmul_kernel.h` | Public API declaration for `my_matmul_kernel` |
| `matmul_kernel.c` | RVV-vectorized (or scalar fallback) compute kernel; **no IREE headers** |
| `matmul_plugin.c` | IREE HAL executable plugin interface (wraps `matmul_kernel.c`) |
| `rvv_standalone_test.c` | Standalone QEMU smoke-test (no IREE dependency) |

### Design Principle

`matmul_kernel.c` is intentionally free of IREE headers, making it usable
for three different build targets without modification:

```
matmul_kernel.c ──┬── (.o)   baked into dispatch ELF at compile time (static)
                  ├── (.so)  IREE plugin loaded via --executable_plugin (dynamic)
                  └──        linked with rvv_standalone_test.c (QEMU unit test)
```

## Kernel ABI

Matches the `func.call` emitted by `MIPSBufferizableOpInterface` after
decomposing 2-D memrefs with `memref.extract_strided_metadata`
(`{llvm.bareptr = true}`, so `memref<f32>` → `float*`):

```c
void my_matmul_kernel(
    const float *A, int64_t A_off, int64_t A_s0, int64_t A_s1,  // lhs [M×K]
    const float *B, int64_t B_off, int64_t B_s0, int64_t B_s1,  // rhs [K×N]
    float       *C, int64_t C_off, int64_t C_s0, int64_t C_s1,  // out [M×N]
    int64_t M, int64_t N, int64_t K);
```

Each 2-D matrix is passed as `(base_ptr, offset, row_stride, col_stride)`,
supporting arbitrary memory layouts (row-major, column-major, non-contiguous).

## Building

Assumes `IREE_SRC` = path to this repo, `CLANG` = `~/miniforge3/bin/clang`.

### Standalone test binary (no IREE, no libc)

```bash
# RISC-V RVV (run under QEMU)
clang --target=riscv64-linux-gnu -march=rv64gcv -mabi=lp64d \
      -O2 -static -nostdlib -ffreestanding -nostdinc \
      -isystem ~/miniforge3/lib/clang/18/include \
      matmul_kernel.c rvv_standalone_test.c -o rvv_test
qemu-riscv64 -cpu rv64,v=true,vlen=512,elen=64,vext_spec=v1.0 ./rvv_test

# x86 host (scalar fallback, with libc)
clang matmul_kernel.c rvv_standalone_test.c -O2 -o rvv_test_host && ./rvv_test_host
```

Expected output:
```
=== rvv_matmul standalone test [RVV]
[1] A * I = A  (4x4 row-major)
[2] 2x3 * 3x2 = 2x2
[3] col-major strides
[4] non-zero base offset
PASSED (20 passed, 0 failed)
```

### Static object (.o) — baked into dispatch ELF

```bash
clang --target=riscv64-linux-gnu -march=rv64gcv -mabi=lp64d \
      -O2 -c -nostdinc -isystem ~/miniforge3/lib/clang/18/include \
      -I "${IREE_SRC}/runtime/src" \
      matmul_kernel.c -o matmul_kernel_riscv.o
```

### Dynamic plugin (.so) — loaded at runtime

```bash
clang --target=riscv64-linux-gnu -march=rv64gcv -mabi=lp64d \
      -O2 -fPIC -shared -nostdinc -nostdlib \
      -isystem ~/miniforge3/lib/clang/18/include \
      -fuse-ld=~/miniforge3/bin/ld.lld \
      -I "${IREE_SRC}/runtime/src" \
      matmul_kernel.c matmul_plugin.c -o librvv_matmul.so
```

## Integration with IREE

```
torch.aten.mm
  ─[ConvertTorchToMIPSPass]──► mips.matmul          (flow IR, tensor domain)
  ─[One-Shot Bufferize]──────► func.call @my_matmul_kernel  (buffers decomposed)
  ─[iree-compile LLVMCPU]────► dispatch ELF inside .vmfb
```

`MIPSBufferizableOpInterface` handles bufferization by decomposing each 2-D
memref into `(base_ptr, offset, stride0, stride1)` via
`memref.extract_strided_metadata` and emitting the `func.call` directly.
No memref form of `mips.matmul` is ever produced in the IR.

### Static Embedding (`--iree-mips-static-embedding`)

Pass `--iree-mips-static-embedding` to `iree-compile`. The bufferizer tags
`my_matmul_kernel` with `{hal.import.static}`, causing the LLVMCPU backend
to emit a direct linker-resolved call. A custom `lld_wrapper.sh` appends
`matmul_kernel_riscv.o` to every dispatch link at compile time.

- No `--executable_plugin` at runtime — kernel is inside the `.vmfb`.
- Requires `-Bsymbolic` in the lld invocation. Without it, lld generates
  `R_RISCV_JUMP_SLOT` in `.rela.plt`; IREE's embedded ELF loader ignores
  `.rela.plt` (only processes `.rela.dyn`), causing a segfault on first call.

### Dynamic Loading (`--executable_plugin`)

Without the flag, `my_matmul_kernel` is a HAL import table entry resolved at
runtime from the plugin `.so` via `iree_hal_executable_plugin_query`.

```bash
iree-run-module --module=matmul.vmfb \
                --executable_plugin=librvv_matmul.so \
                --function=matmul_4x4 ...
```

## End-to-End Workflow Scripts

See [`build_tools/riscv/`](../../../../../build_tools/riscv/) in the repo root:

| Script | Description |
|--------|-------------|
| `setup_qemu_workflow.sh` | One-time setup: toolchain, QEMU, IREE host + RISC-V builds |
| `rvv_qemu_workflow_static.sh` | Static-embedding pipeline + QEMU run |
| `rvv_qemu_workflow_dynamic.sh` | Dynamic-plugin pipeline + QEMU run |
