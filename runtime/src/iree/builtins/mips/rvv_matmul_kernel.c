// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RVV (RISC-V Vector 1.0) matmul kernel exposed as an IREE executable plugin.
//
// ABI contract (must match MIPSBufferizableOpInterface.cpp / decomposeMemref2D):
//   Each 2-D memref is decomposed into (base_ptr, offset, stride0, stride1).
//   {llvm.bareptr = true} means memref<f32> → float* in C.
//   index → int64_t on RV64.
//
// C signature (15 args total):
//   void my_matmul_kernel(
//     float* A, int64_t A_off, int64_t A_s0, int64_t A_s1,   // lhs [M×K]
//     float* B, int64_t B_off, int64_t B_s0, int64_t B_s1,   // rhs [K×N]
//     float* C, int64_t C_off, int64_t C_s0, int64_t C_s1,   // out [M×N]
//     int64_t M, int64_t N, int64_t K
//   );
//
// Vectorization strategy (RVV LMUL=m4):
//   - Outer loops: m (rows of A) and k (contraction axis)
//   - Inner loop : n (columns of B), vectorized with vsetvl_e32m4
//   - Each vl-wide chunk of C[m, n:n+vl] is kept in a vfloat32m4_t accumulator
//     and accumulated across the full k-dimension before storing.
//   - Handles arbitrary strides via conditional vlse/vsse.

#include <stddef.h>
#include <stdint.h>

#include "iree/hal/local/executable_plugin.h"

#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

//===----------------------------------------------------------------------===//
// Core RVV kernel
//===----------------------------------------------------------------------===//

// The kernel is separated from the plugin wrapper so it can be unit-tested
// with a plain C harness (rvv_standalone_test.c) without IREE headers.
//
// For non-RV targets (e.g. when the host compiler processes this file) the
// function falls back to a scalar triple loop so the plugin still links.

#ifdef __riscv_vector

// RVV implementation — compiled only when targeting RV64GCV.
static void rvv_matmul_core(
    const float *A, int64_t A_off, int64_t A_s0, int64_t A_s1,
    const float *B, int64_t B_off, int64_t B_s0, int64_t B_s1,
    float *C, int64_t C_off, int64_t C_s0, int64_t C_s1,
    int64_t M, int64_t N, int64_t K) {
  // Apply offsets once — identical to the scalar implementation.
  A += A_off;
  B += B_off;
  C += C_off;

  // Detect unit-stride fast paths at runtime to choose vle/vlse.
  const int a_unit_col = (A_s1 == 1);
  const int b_unit_col = (B_s1 == 1);
  const int c_unit_col = (C_s1 == 1);

  for (int64_t m = 0; m < M; ++m) {
    int64_t n = 0;
    while (n < N) {
      // Set vector length for this strip of N columns.
      size_t vl = __riscv_vsetvl_e32m4((size_t)(N - n));

      // Zero accumulator — one vfloat32m4_t per output strip C[m, n:n+vl].
      vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

      for (int64_t k = 0; k < K; ++k) {
        // Scalar load: A[m, k] (row-major access — s0=N, s1=1 for row-major).
        float a_val;
        if (a_unit_col) {
          a_val = A[m * A_s0 + k];
        } else {
          a_val = A[m * A_s0 + k * A_s1];
        }

        // Vector load: B[k, n:n+vl].
        vfloat32m4_t b_vec;
        if (b_unit_col) {
          b_vec = __riscv_vle32_v_f32m4(&B[k * B_s0 + n], vl);
        } else {
          b_vec = __riscv_vlse32_v_f32m4(
              &B[k * B_s0 + n * B_s1],
              (ptrdiff_t)(B_s1 * (int64_t)sizeof(float)), vl);
        }

        // acc += a_val * b_vec
        acc = __riscv_vfmacc_vf_f32m4(acc, a_val, b_vec, vl);
      }

      // Store accumulator to C[m, n:n+vl].
      if (c_unit_col) {
        __riscv_vse32_v_f32m4(&C[m * C_s0 + n], acc, vl);
      } else {
        __riscv_vsse32_v_f32m4(
            &C[m * C_s0 + n * C_s1],
            (ptrdiff_t)(C_s1 * (int64_t)sizeof(float)), acc, vl);
      }
      n += (int64_t)vl;
    }
  }
}

#else // !__riscv_vector

// Scalar fallback — used when this file is compiled for the host (x86/arm)
// during CI or initial bring-up.
static void rvv_matmul_core(
    const float *A, int64_t A_off, int64_t A_s0, int64_t A_s1,
    const float *B, int64_t B_off, int64_t B_s0, int64_t B_s1,
    float *C, int64_t C_off, int64_t C_s0, int64_t C_s1,
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
// Direct-call entry point for static embedding
//===----------------------------------------------------------------------===//
// When the dispatch ELF is built with --iree-llvmcpu-link-embedded=true and
// the function declaration carries {hal.import.static}, LLVMCPU codegen emits
// a direct call to this symbol instead of going through the HAL import table.
// The lld wrapper appends this .o to the linker invocation so the symbol is
// resolved at link time.

void my_matmul_kernel(
    const float *A, int64_t A_off, int64_t A_s0, int64_t A_s1,
    const float *B, int64_t B_off, int64_t B_s0, int64_t B_s1,
    float *C, int64_t C_off, int64_t C_s0, int64_t C_s1,
    int64_t M, int64_t N, int64_t K) {
  rvv_matmul_core(A, A_off, A_s0, A_s1,
                  B, B_off, B_s0, B_s1,
                  C, C_off, C_s0, C_s1,
                  M, N, K);
}

//===----------------------------------------------------------------------===//
// IREE Executable Plugin interface
//===----------------------------------------------------------------------===//

// Packed argument struct — mirrors the func.call argument list emitted by
// MIPSBufferizableOpInterface::bufferize() → decomposeMemref2D().
typedef struct {
  float *A;
  int64_t A_off, A_s0, A_s1;
  float *B;
  int64_t B_off, B_s0, B_s1;
  float *C;
  int64_t C_off, C_s0, C_s1;
  int64_t M, N, K;
} rvv_matmul_kernel_args_t;

static int rvv_matmul_kernel_import(void *params_ptr, void *context,
                                    void *reserved) {
  (void)context;
  (void)reserved;
  const rvv_matmul_kernel_args_t *a =
      (const rvv_matmul_kernel_args_t *)params_ptr;
  rvv_matmul_core(a->A, a->A_off, a->A_s0, a->A_s1,
                  a->B, a->B_off, a->B_s0, a->B_s1,
                  a->C, a->C_off, a->C_s0, a->C_s1,
                  a->M, a->N, a->K);
  return 0;
}

static iree_hal_executable_plugin_status_t plugin_load(
    const iree_hal_executable_plugin_environment_v0_t *environment,
    size_t param_count,
    const iree_hal_executable_plugin_string_pair_t *params, void **out_self) {
  (void)environment;
  (void)param_count;
  (void)params;
  *out_self = NULL;
  return iree_hal_executable_plugin_ok_status();
}

static void plugin_unload(void *self) { (void)self; }

static iree_hal_executable_plugin_status_t plugin_resolve(
    void *self, const iree_hal_executable_plugin_resolve_params_v0_t *params,
    iree_hal_executable_plugin_resolution_t *out_resolution) {
  (void)self;
  *out_resolution = 0;
  bool any_required_not_found = false;

  for (size_t i = 0; i < params->count; ++i) {
    if (params->out_fn_ptrs[i]) continue;
    const char *name = params->symbol_names[i];
    bool optional = iree_hal_executable_plugin_import_is_optional(name);
    if (optional) ++name;

    if (iree_hal_executable_plugin_strcmp(name, "my_matmul_kernel") == 0) {
      params->out_fn_ptrs[i] = rvv_matmul_kernel_import;
      params->out_fn_contexts[i] = NULL;
    } else {
      if (!optional) any_required_not_found = true;
    }
  }

  return any_required_not_found
             ? iree_hal_executable_plugin_status_from_code(
                   IREE_HAL_EXECUTABLE_PLUGIN_STATUS_NOT_FOUND)
             : iree_hal_executable_plugin_ok_status();
}

IREE_HAL_EXECUTABLE_PLUGIN_EXPORT const iree_hal_executable_plugin_header_t **
iree_hal_executable_plugin_query(
    iree_hal_executable_plugin_version_t max_version, void *reserved) {
  static const iree_hal_executable_plugin_header_t header = {
      .version = IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST,
      .name = "rvv_matmul",
      .description = "RISC-V RVV 1.0 matmul kernel plugin",
      .features = IREE_HAL_EXECUTABLE_PLUGIN_FEATURE_STANDALONE,
      .sanitizer = IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_KIND,
  };
  static const iree_hal_executable_plugin_v0_t plugin = {
      .header = &header,
      .load = plugin_load,
      .unload = plugin_unload,
      .resolve = plugin_resolve,
  };
  return max_version <= IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST
             ? (const iree_hal_executable_plugin_header_t **)&plugin
             : NULL;
}
