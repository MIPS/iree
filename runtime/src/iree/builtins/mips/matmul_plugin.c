// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// IREE Executable Plugin interface for the MIPS matmul kernel.
//
// This file wires my_matmul_kernel into the IREE HAL plugin ABI so the
// function can be resolved at runtime via --executable_plugin.
//
// Build as a shared library alongside matmul_kernel.c:
//   clang --target=riscv64-linux-gnu -march=rv64gcv -mabi=lp64d \
//         -O2 -fPIC -shared -nostdinc ... \
//         matmul_kernel.c matmul_plugin.c -o librvv_matmul.so

#include "matmul_kernel.h"

#include "iree/hal/local/executable_plugin.h"

//===----------------------------------------------------------------------===//
// Import wrapper
//===----------------------------------------------------------------------===//
// The HAL plugin dispatch table expects functions with the signature:
//   int fn(void *params_ptr, void *context, void *reserved)
// where params_ptr points to a packed struct matching the func.call ABI
// emitted by MIPSBufferizableOpInterface::bufferize() → decomposeMemref2D().

typedef struct {
  float   *A;
  int64_t  A_off, A_s0, A_s1;
  float   *B;
  int64_t  B_off, B_s0, B_s1;
  float   *C;
  int64_t  C_off, C_s0, C_s1;
  int64_t  M, N, K;
} matmul_kernel_args_t;

static int matmul_kernel_import(void *params_ptr, void *context,
                                void *reserved) {
  (void)context;
  (void)reserved;
  const matmul_kernel_args_t *a = (const matmul_kernel_args_t *)params_ptr;
  my_matmul_kernel(a->A, a->A_off, a->A_s0, a->A_s1,
                   a->B, a->B_off, a->B_s0, a->B_s1,
                   a->C, a->C_off, a->C_s0, a->C_s1,
                   a->M, a->N, a->K);
  return 0;
}

//===----------------------------------------------------------------------===//
// Plugin lifecycle
//===----------------------------------------------------------------------===//

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
      params->out_fn_ptrs[i] = matmul_kernel_import;
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

//===----------------------------------------------------------------------===//
// Plugin query entry point
//===----------------------------------------------------------------------===//

IREE_HAL_EXECUTABLE_PLUGIN_EXPORT const iree_hal_executable_plugin_header_t **
iree_hal_executable_plugin_query(
    iree_hal_executable_plugin_version_t max_version, void *reserved) {
  static const iree_hal_executable_plugin_header_t header = {
      .version     = IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST,
      .name        = "mips_matmul",
      .description = "RISC-V RVV 1.0 matmul kernel plugin",
      .features    = IREE_HAL_EXECUTABLE_PLUGIN_FEATURE_STANDALONE,
      .sanitizer   = IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_KIND,
  };
  static const iree_hal_executable_plugin_v0_t plugin = {
      .header  = &header,
      .load    = plugin_load,
      .unload  = plugin_unload,
      .resolve = plugin_resolve,
  };
  return max_version <= IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST
             ? (const iree_hal_executable_plugin_header_t **)&plugin
             : NULL;
}
