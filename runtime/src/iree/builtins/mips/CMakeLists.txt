# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Shared library containing the MIPS custom matmul kernel.
#
# matmul_kernel.c — RVV (or scalar fallback) compute kernel; no IREE headers.
# matmul_plugin.c — IREE HAL executable plugin interface.
#
# The plugin is loaded at runtime via:
#   iree-run-module --executable_plugin=libmips_matmul.so ...

add_library(mips_matmul SHARED
  matmul_kernel.c
  matmul_plugin.c
)

target_include_directories(mips_matmul
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE
    # For iree/hal/local/executable_plugin.h (standalone C header, no deps)
    ${PROJECT_SOURCE_DIR}/runtime/src
)

set_target_properties(mips_matmul PROPERTIES
  C_VISIBILITY_PRESET default
  POSITION_INDEPENDENT_CODE ON
)
