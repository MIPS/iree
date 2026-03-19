// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RVV (RISC-V Vector 1.0) f32 matmul kernel.
//
// This file is intentionally free of IREE headers so it can be:
//   - Compiled to a .o and baked into the dispatch ELF (static embedding).
//   - Linked into the IREE plugin .so alongside matmul_plugin.c.
//   - Compiled standalone for unit tests (rvv_standalone_test.c).
//
// Vectorization strategy (RVV LMUL=m4):
//   Outer loops: m (rows of A) and k (contraction axis).
//   Inner loop : n (columns of B), vectorized with vsetvl_e32m4.
//   Each vl-wide strip of C[m, n:n+vl] is accumulated across k before storing.
//   Arbitrary strides handled via conditional vlse/vsse vs vle/vse.

#include "matmul_kernel.h"

#include <stddef.h>

#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

//===----------------------------------------------------------------------===//
// Internal compute kernel
//===----------------------------------------------------------------------===//

#ifdef __riscv_vector

static void rvv_matmul_core(
    const float *A, int64_t A_off, int64_t A_s0, int64_t A_s1,
    const float *B, int64_t B_off, int64_t B_s0, int64_t B_s1,
    float       *C, int64_t C_off, int64_t C_s0, int64_t C_s1,
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
      size_t vl = __riscv_vsetvl_e32m4((size_t)(N - n));
      vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

      for (int64_t k = 0; k < K; ++k) {
        float a_val = a_unit_col ? A[m * A_s0 + k]
                                 : A[m * A_s0 + k * A_s1];
        vfloat32m4_t b_vec =
            b_unit_col
                ? __riscv_vle32_v_f32m4(&B[k * B_s0 + n], vl)
                : __riscv_vlse32_v_f32m4(
                      &B[k * B_s0 + n * B_s1],
                      (ptrdiff_t)(B_s1 * (int64_t)sizeof(float)), vl);
        acc = __riscv_vfmacc_vf_f32m4(acc, a_val, b_vec, vl);
      }

      if (c_unit_col)
        __riscv_vse32_v_f32m4(&C[m * C_s0 + n], acc, vl);
      else
        __riscv_vsse32_v_f32m4(
            &C[m * C_s0 + n * C_s1],
            (ptrdiff_t)(C_s1 * (int64_t)sizeof(float)), acc, vl);
      n += (int64_t)vl;
    }
  }
}

#else // scalar fallback

static void rvv_matmul_core(
    const float *A, int64_t A_off, int64_t A_s0, int64_t A_s1,
    const float *B, int64_t B_off, int64_t B_s0, int64_t B_s1,
    float       *C, int64_t C_off, int64_t C_s0, int64_t C_s1,
    int64_t M, int64_t N, int64_t K) {
  A += A_off;
  B += B_off;
  C += C_off;
  for (int64_t m = 0; m < M; ++m)
    for (int64_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k)
        acc += A[m * A_s0 + k * A_s1] * B[k * B_s0 + n * B_s1];
      C[m * C_s0 + n * C_s1] = acc;
    }
}

#endif // __riscv_vector

//===----------------------------------------------------------------------===//
// Public entry point
//===----------------------------------------------------------------------===//

void my_matmul_kernel(
    const float *A, int64_t A_off, int64_t A_s0, int64_t A_s1,
    const float *B, int64_t B_off, int64_t B_s0, int64_t B_s1,
    float       *C, int64_t C_off, int64_t C_s0, int64_t C_s1,
    int64_t M, int64_t N, int64_t K) {
  rvv_matmul_core(A, A_off, A_s0, A_s1,
                  B, B_off, B_s0, B_s1,
                  C, C_off, C_s0, C_s1,
                  M, N, K);
}
