// mips_matmul_test.mlir
//
// End-to-end test inputs for the MIPS matmul kernel pipeline.
//
// Each function exercises torch.aten.mm, which is intercepted by
// ConvertTorchToMIPSPass and rewritten as mips.matmul. The op is then
// eliminated during One-Shot Bufferize: MIPSBufferizableOpInterface
// decomposes the 2-D memrefs and emits a direct func.call to the
// hand-tuned C kernel:
//
//   torch.aten.mm
//     → mips.matmul          (ConvertTorchToMIPSPass)
//     → flow.dispatch(...)   (IREE dispatch formation)
//     → func.call @my_matmul_kernel  (MIPSBufferizableOpInterface)
//     → ELF inside .vmfb     (iree-compile LLVMCPU backend)
//
// Usage:
//   bash build_tools/riscv/rvv_qemu_workflow_static.sh  -- static  (.o baked into vmfb)
//   bash build_tools/riscv/rvv_qemu_workflow_dynamic.sh -- dynamic (.so plugin at runtime)

module {
  // ── Test 1: 4×4 identity × data → passthrough ────────────────────────────
  // Verifies that A=I leaves B unchanged; a simple correctness smoke-test.
  //
  // A = identity(4×4), B = [[1..4],[5..8],[9..12],[13..16]]
  // Expected: result = B
  func.func @matmul_4x4(
      %A : !torch.vtensor<[4,4],f32>,
      %B : !torch.vtensor<[4,4],f32>)
      -> !torch.vtensor<[4,4],f32> {
    %0 = torch.aten.mm %A, %B
        : !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,4],f32>
        -> !torch.vtensor<[4,4],f32>
    return %0 : !torch.vtensor<[4,4],f32>
  }

  // ── Test 2: 2×3 × 3×2 → 2×2  (non-square, reduced K dimension) ──────────
  // Verifies M≠N≠K path through the kernel (inner loop trip-count < vlen).
  //
  // A = [[1,2,3],[4,5,6]], B = [[1,0],[0,1],[1,0]]
  // Expected: [[1+0+3, 0+2+0],[4+0+6, 0+5+0]] = [[4,2],[10,5]]
  func.func @matmul_2x3x2(
      %A : !torch.vtensor<[2,3],f32>,
      %B : !torch.vtensor<[3,2],f32>)
      -> !torch.vtensor<[2,2],f32> {
    %0 = torch.aten.mm %A, %B
        : !torch.vtensor<[2,3],f32>, !torch.vtensor<[3,2],f32>
        -> !torch.vtensor<[2,2],f32>
    return %0 : !torch.vtensor<[2,2],f32>
  }

  // ── Test 3: 8×8 × 8×8 → 8×8  (exercises multi-vector-register tiling) ───
  // With vlen=512 and LMUL=m4, N=8 fits in a single VL group. This test
  // stresses the vectorized inner loop and accumulation across K=8 steps.
  //
  // A = upper-triangular ones (row i has ones in columns 0..i).
  // B = identity(8×8).
  // Expected: A*I = A — result is upper-triangular ones.
  //
  // A row layout (8×8):
  //   row 0: [1,0,0,0,0,0,0,0]
  //   row 1: [1,1,0,0,0,0,0,0]
  //   row 2: [1,1,1,0,0,0,0,0]
  //   ...
  //   row 7: [1,1,1,1,1,1,1,1]
  func.func @matmul_8x8(
      %A : !torch.vtensor<[8,8],f32>,
      %B : !torch.vtensor<[8,8],f32>)
      -> !torch.vtensor<[8,8],f32> {
    %0 = torch.aten.mm %A, %B
        : !torch.vtensor<[8,8],f32>, !torch.vtensor<[8,8],f32>
        -> !torch.vtensor<[8,8],f32>
    return %0 : !torch.vtensor<[8,8],f32>
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Expected outputs (iree-run-module)
// ─────────────────────────────────────────────────────────────────────────────
//
// matmul_4x4  A=identity(4x4), B=[1..16 row-major]:
//   result[0]: 4x4xf32=[1 2 3 4][5 6 7 8][9 10 11 12][13 14 15 16]
//
// matmul_2x3x2  A=[[1,2,3],[4,5,6]], B=[[1,0],[0,1],[1,0]]:
//   result[0]: 2x2xf32=[4 2][10 5]
//
// matmul_8x8  A=upper-triangular-ones(8x8), B=identity(8x8):
//   result[0]: 8x8xf32=
//     [1 0 0 0 0 0 0 0]
//     [1 1 0 0 0 0 0 0]
//     [1 1 1 0 0 0 0 0]
//     [1 1 1 1 0 0 0 0]
//     [1 1 1 1 1 0 0 0]
//     [1 1 1 1 1 1 0 0]
//     [1 1 1 1 1 1 1 0]
//     [1 1 1 1 1 1 1 1]
//
// iree-run-module invocation for matmul_8x8:
//   --function=matmul_8x8
//   "--input=8x8xf32=1,0,0,0,0,0,0,0, 1,1,0,0,0,0,0,0, 1,1,1,0,0,0,0,0, 1,1,1,1,0,0,0,0, 1,1,1,1,1,0,0,0, 1,1,1,1,1,1,0,0, 1,1,1,1,1,1,1,0, 1,1,1,1,1,1,1,1"
//   "--input=8x8xf32=1,0,0,0,0,0,0,0, 0,1,0,0,0,0,0,0, 0,0,1,0,0,0,0,0, 0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0, 0,0,0,0,0,1,0,0, 0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1"
