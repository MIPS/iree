// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RVV (RISC-V Vector 1.0) INT8 matmul kernel: C[i32] = A[i8] * B[i8].
//
// This file is intentionally free of IREE headers so it can be:
//   - Compiled to a .o and baked into the dispatch ELF (static embedding).
//   - Linked into the IREE plugin .so alongside matmul_plugin.c.
//   - Compiled standalone for unit tests.
//
// Vectorization strategy (RVV widening multiply):
//   Outer loops: m (rows of A) and k (contraction axis).
//   Inner loop : n (columns of B), vectorized.
//
//   LMUL selection to match widening chain:
//     i8  LMUL=m2  → VLMAX = (VLEN/8)  * 2 = VLEN/4
//     i16 LMUL=m4  → VLMAX = (VLEN/16) * 4 = VLEN/4  (after first widening)
//     i32 LMUL=m8  → VLMAX = (VLEN/32) * 8 = VLEN/4  (accumulator)
//   All three LMULs yield the same VLMAX, so the same application vl is valid
//   for all three element widths — no secondary vsetvl is needed per intrinsic.
//
//   Per n-strip:
//     acc[vl] (i32m8) = 0
//     for k in 0..K:
//       a_val (i8 scalar) = A[m, k]
//       b_i8  (i8m2  vec) = B[k, n:n+vl]
//       b_i16 (i16m4 vec) = sign_extend(b_i8)
//       acc              += widen_macc(a_val, b_i16)   // i16 * i16 → i32
//     C[m, n:n+vl] = acc
//
// The widening chain avoids intermediate overflow:
//   i8 × i8 → i16 intermediate → accumulated into i32.

#include "matmul_kernel.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

//===----------------------------------------------------------------------===//
// Internal compute kernel (RVV path)
//===----------------------------------------------------------------------===//

#ifdef __riscv_vector

static void rvv_matmul_i8_core(
    const int8_t *A, int64_t A_off, int64_t A_s0, int64_t A_s1,
    const int8_t *B, int64_t B_off, int64_t B_s0, int64_t B_s1,
    int32_t      *C, int64_t C_off, int64_t C_s0, int64_t C_s1,
    int64_t M, int64_t N, int64_t K) {
  A += A_off;
  B += B_off;
  C += C_off;

  const int a_unit_col = (A_s1 == 1);
  const int b_unit_col = (B_s1 == 1);
  const int c_unit_col = (C_s1 == 1);

  for (int64_t m = 0; m < M; ++m) {
    int64_t n = 0;
    while (n < N) {
      // vl: element count for this n-strip.
      // __riscv_vsetvl_e32m8 caps vl at VLMAX_i32m8 = VLEN/4, which equals
      // VLMAX_i8m2 and VLMAX_i16m4 — all intrinsics below share this vl.
      size_t vl = __riscv_vsetvl_e32m8((size_t)(N - n));

      // Initialize i32 accumulator to zero.
      vint32m8_t acc = __riscv_vmv_v_x_i32m8(0, vl);

      for (int64_t k = 0; k < K; ++k) {
        // Scalar load from A[m, k].
        int8_t a_val = a_unit_col ? A[m * A_s0 + k]
                                  : A[m * A_s0 + k * A_s1];

        // Vector load B[k, n:n+vl] as i8.
        vint8m2_t b_i8 =
            b_unit_col
                ? __riscv_vle8_v_i8m2((const int8_t *)&B[k * B_s0 + n], vl)
                : __riscv_vlse8_v_i8m2(
                      (const int8_t *)&B[k * B_s0 + n * B_s1],
                      (ptrdiff_t)(B_s1 * (int64_t)sizeof(int8_t)), vl);

        // Sign-extend i8m2 → i16m4 (same element count, double the width).
        vint16m4_t b_i16 = __riscv_vsext_vf2_i16m4(b_i8, vl);

        // Widening signed multiply-accumulate:
        //   acc[i] += sign_ext_32(a_val) * sign_ext_32(b_i16[i])
        // vwmacc.vx  vd[i32m8],  rs1[i16],  vs2[i16m4]
        acc = __riscv_vwmacc_vx_i32m8(acc, (int16_t)a_val, b_i16, vl);
      }

      // Store accumulator to C[m, n:n+vl].
      if (c_unit_col)
        __riscv_vse32_v_i32m8((int32_t *)&C[m * C_s0 + n], acc, vl);
      else
        __riscv_vsse32_v_i32m8(
            (int32_t *)&C[m * C_s0 + n * C_s1],
            (ptrdiff_t)(C_s1 * (int64_t)sizeof(int32_t)), acc, vl);

      n += (int64_t)vl;
    }
  }
}

#else // scalar fallback

static void rvv_matmul_i8_core(
    const int8_t *A, int64_t A_off, int64_t A_s0, int64_t A_s1,
    const int8_t *B, int64_t B_off, int64_t B_s0, int64_t B_s1,
    int32_t      *C, int64_t C_off, int64_t C_s0, int64_t C_s1,
    int64_t M, int64_t N, int64_t K) {
  A += A_off;
  B += B_off;
  C += C_off;
  for (int64_t m = 0; m < M; ++m)
    for (int64_t n = 0; n < N; ++n) {
      int32_t acc = 0;
      for (int64_t k = 0; k < K; ++k)
        acc += (int32_t)A[m * A_s0 + k * A_s1] *
               (int32_t)B[k * B_s0 + n * B_s1];
      C[m * C_s0 + n * C_s1] = acc;
    }
}

#endif // __riscv_vector

//===----------------------------------------------------------------------===//
// Public entry point
//===----------------------------------------------------------------------===//

void my_matmul_kernel_i8(
    const int8_t *A, int64_t A_off, int64_t A_s0, int64_t A_s1,
    const int8_t *B, int64_t B_off, int64_t B_s0, int64_t B_s1,
    int32_t      *C, int64_t C_off, int64_t C_s0, int64_t C_s1,
    int64_t M, int64_t N, int64_t K) {
  rvv_matmul_i8_core(A, A_off, A_s0, A_s1,
                     B, B_off, B_s0, B_s1,
                     C, C_off, C_s0, C_s1,
                     M, N, K);
}
