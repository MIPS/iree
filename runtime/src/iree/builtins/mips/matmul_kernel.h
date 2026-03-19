// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Public API for the MIPS matmul kernel.
//
// ABI contract (must match MIPSBufferizableOpInterface.cpp / decomposeMemref2D):
//   Each 2-D memref is decomposed into (base_ptr, offset, stride0, stride1).
//   {llvm.bareptr = true} means memref<f32> → float* in C.
//   index → int64_t on RV64.
//
// C signatures (15 args each):
//   void my_matmul_kernel(
//     float* A, int64_t A_off, int64_t A_s0, int64_t A_s1,   // lhs [M×K]
//     float* B, int64_t B_off, int64_t B_s0, int64_t B_s1,   // rhs [K×N]
//     float* C, int64_t C_off, int64_t C_s0, int64_t C_s1,   // out [M×N]
//     int64_t M, int64_t N, int64_t K
//   );
//   void my_matmul_kernel_i8(
//     int8_t*  A, int64_t A_off, int64_t A_s0, int64_t A_s1, // lhs [M×K] i8
//     int8_t*  B, int64_t B_off, int64_t B_s0, int64_t B_s1, // rhs [K×N] i8
//     int32_t* C, int64_t C_off, int64_t C_s0, int64_t C_s1, // out [M×N] i32
//     int64_t M, int64_t N, int64_t K
//   );

#ifndef IREE_BUILTINS_MIPS_MATMUL_KERNEL_H_
#define IREE_BUILTINS_MIPS_MATMUL_KERNEL_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 2-D f32 matmul: C = A * B, Destination-Passing Style.
// Supports arbitrary row/col strides and non-zero base offsets.
// RVV-vectorized on RISC-V targets; scalar fallback elsewhere.
void my_matmul_kernel(
    const float *A, int64_t A_off, int64_t A_s0, int64_t A_s1,
    const float *B, int64_t B_off, int64_t B_s0, int64_t B_s1,
    float       *C, int64_t C_off, int64_t C_s0, int64_t C_s1,
    int64_t M, int64_t N, int64_t K);

// 2-D INT8 matmul: C[M×N] (i32) = A[M×K] (i8) * B[K×N] (i8).
// C is zero-initialized by the caller before the call.
// Supports arbitrary row/col strides and non-zero base offsets.
// RVV-vectorized (widening i8→i16→i32) on RISC-V; scalar fallback elsewhere.
void my_matmul_kernel_i8(
    const int8_t *A, int64_t A_off, int64_t A_s0, int64_t A_s1,
    const int8_t *B, int64_t B_off, int64_t B_s0, int64_t B_s1,
    int32_t      *C, int64_t C_off, int64_t C_s0, int64_t C_s1,
    int64_t M, int64_t N, int64_t K);

#ifdef __cplusplus
}
#endif

#endif // IREE_BUILTINS_MIPS_MATMUL_KERNEL_H_
