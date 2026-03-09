// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_MIPS_BASE
#define IREE_DIALECT_MIPS_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// MIPS dialect definition
//===----------------------------------------------------------------------===//

def MIPS_Dialect : Dialect {
  let name = "mips";
  let cppNamespace = "::mlir::iree_compiler::IREE::MIPS";
  let summary = "Semantic dialect layer for dispatching to highly-optimized MIPS kernels.";
  let description = [{
    The MIPS dialect provides a semantic abstraction layer that bridges
    high-level tensor operations and hand-tuned, target-specific kernel
    implementations (e.g. RVV-vectorized routines).

    Ops in this dialect live entirely in the tensor domain — no memref forms
    are ever produced. They are eliminated during One-Shot Bufferize:
    the BufferizableOpInterface implementation allocates output buffers and
    emits direct `func.call` instructions to the underlying C kernels.

    The semantic level allows the compiler to apply higher-level
    transformations before lowering, such as:
      - Fusing producer ops (e.g. transposes, type promotions) into a single
        kernel call to avoid intermediate allocations.
      - Tiling and packing decisions based on target vector width.
      - Selecting between kernel variants (e.g. RVV LMUL=m4 vs. m8) based
        on operand shapes or hardware configuration.

    Pipeline:
      torch.aten.mm → mips.matmul → func.call @my_matmul_kernel
  }];

  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::tensor::TensorDialect"
  ];

  // No custom attribute types → do not declare parseAttribute/printAttribute
  // overrides. The base Dialect class handles the fallback behavior.
  let useDefaultAttributePrinterParser = 0;
}

//===----------------------------------------------------------------------===//
// Base op class
//===----------------------------------------------------------------------===//

class MIPS_Op<string mnemonic, list<Trait> traits = []>
    : Op<MIPS_Dialect, mnemonic, traits>;

#endif // IREE_DIALECT_MIPS_BASE
