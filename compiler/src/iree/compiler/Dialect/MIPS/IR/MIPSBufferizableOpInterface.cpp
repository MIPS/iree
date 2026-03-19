// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements BufferizableOpInterface for mips.matmul.
//
// mips.matmul is a tensor-only, Destination-Passing-Style (DPS) op. It is
// eliminated *entirely* during One-Shot Bufferize: bufferize() obtains memref
// buffers for all three operands, decomposes each 2-D memref into
// (base_ptr, offset, stride0, stride1) via memref.extract_strided_metadata,
// and emits a func.call to the appropriate kernel directly.
//
// The kernel is selected based on the LHS element type:
//   f32  → func.call @my_matmul_kernel      (f32 × f32 → f32)
//   i8   → func.call @my_matmul_kernel_i8   (i8  × i8  → i32)
//
// No memref form of mips.matmul ever exists in the IR.
//
// Before bufferization (f32 example):
//   %C = mips.matmul %A, %B, %init
//       : tensor<MxKxf32>, tensor<KxNxf32>, tensor<MxNxf32> -> tensor<MxNxf32>
//
// After bufferization (produced inside bufferize()):
//   %A_meta  = memref.extract_strided_metadata %A_buf  -> (base, off, s0, s1)
//   %B_meta  = memref.extract_strided_metadata %B_buf  -> (base, off, s0, s1)
//   %C_meta  = memref.extract_strided_metadata %C_buf  -> (base, off, s0, s1)
//   %M = memref.dim %A_buf, 0
//   %N = memref.dim %B_buf, 1
//   %K = memref.dim %A_buf, 1
//   call @my_matmul_kernel(%A_base, %A_off, %A_s0, %A_s1,
//                          %B_base, %B_off, %B_s0, %B_s1,
//                          %C_base, %C_off, %C_s0, %C_s1,
//                          %M, %N, %K)
//   -- tensor result replaced by %C_buf via replaceOpWithBufferizedValues --
//
// The INT8 path is identical but uses memref<i8> / memref<i32> base pointers
// and calls @my_matmul_kernel_i8 instead.

#include "iree/compiler/Dialect/MIPS/IR/MIPSDialect.h"
#include "iree/compiler/Dialect/MIPS/IR/MIPSOps.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace mlir::bufferization;

// When true, matmul kernels are emitted as direct linker-resolved calls
// (hal.import.static) instead of dynamic HAL import table entries.
// Pass --iree-mips-static-embedding to iree-compile to enable.
// Mutually exclusive with --executable_plugin at runtime.
static llvm::cl::opt<bool> clMIPSStaticEmbedding(
    "iree-mips-static-embedding",
    llvm::cl::desc(
        "Emit mips matmul kernels as direct linker-resolved calls "
        "(hal.import.static) instead of dynamic HAL imports. "
        "Requires the kernel .o to be appended by lld_wrapper at compile "
        "time. Mutually exclusive with --executable_plugin at runtime."),
    llvm::cl::init(false));

namespace mlir::iree_compiler::IREE::MIPS {
namespace {

static constexpr StringLiteral kKernelF32 = "my_matmul_kernel";
static constexpr StringLiteral kKernelI8  = "my_matmul_kernel_i8";

//===----------------------------------------------------------------------===//
// Helper: ensure func.func private @<kernelName> exists at module scope.
//
// The declaration carries {llvm.bareptr = true} so the LLVM backend passes
// bare pointer arguments instead of MLIR memref descriptor structs, matching
// the C kernel ABI.
//===----------------------------------------------------------------------===//

static func::FuncOp ensureKernelDeclaration(RewriterBase &rewriter,
                                             Operation *moduleOp,
                                             StringRef kernelName,
                                             FunctionType fnType,
                                             Location loc) {
  if (auto existing = dyn_cast_if_present<func::FuncOp>(
          SymbolTable::lookupSymbolIn(moduleOp, kernelName)))
    return existing;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&moduleOp->getRegion(0).front());
  auto fnDecl = func::FuncOp::create(rewriter, loc, kernelName, fnType);
  SymbolTable::setSymbolVisibility(fnDecl, SymbolTable::Visibility::Private);
  fnDecl->setAttr("llvm.bareptr", rewriter.getBoolAttr(true));
  // If --iree-mips-static-embedding was passed, emit a direct linker call
  // instead of a dynamic HAL import table entry.
  if (clMIPSStaticEmbedding)
    fnDecl->setAttr("hal.import.static", rewriter.getUnitAttr());
  return fnDecl;
}

//===----------------------------------------------------------------------===//
// Helper: decompose a 2-D memref into (base_ptr, offset, stride0, stride1).
//
// Uses memref.extract_strided_metadata.  The base_ptr is always a rank-0
// memref with DEFAULT address space (memref<elem_type>), regardless of the
// source memref's address space.  Any IREE-specific memory space (e.g.
// #hal.descriptor_type<storage_buffer>) is stripped via
// memref.memory_space_cast so that:
//
//   1. The function declaration uses plain memref<elem_type>, which is stable
//      across all pipeline stages.
//   2. eraseHALDescriptorTypeFromMemRefPass (which runs after bufferization and
//      does NOT update external function declarations) cannot introduce a
//      type mismatch between the call operands and the declaration.
//
// Combined with the {llvm.bareptr = true} attribute on the callee, the
// rank-0 memref<elem_type> lowers to a bare pointer matching the C ABI.
//===----------------------------------------------------------------------===//

static void decomposeMemref2D(RewriterBase &rewriter, Location loc,
                               Value memref2D,
                               SmallVectorImpl<Value> &callOperands,
                               SmallVectorImpl<Type> &callArgTypes) {
  Type indexType = IndexType::get(rewriter.getContext());

  auto meta =
      memref::ExtractStridedMetadataOp::create(rewriter, loc, memref2D);

  // Strip any IREE-specific memory space from the base pointer so the
  // function declaration stays in the default address space.
  Value basePtr = meta.getBaseBuffer();
  auto basePtrMemrefTy = cast<MemRefType>(basePtr.getType());
  MemRefType plainBasePtrTy =
      MemRefType::get(/*shape=*/{}, basePtrMemrefTy.getElementType());
  if (basePtrMemrefTy != plainBasePtrTy) {
    basePtr = memref::MemorySpaceCastOp::create(rewriter, loc, plainBasePtrTy,
                                                basePtr);
  }

  callOperands.push_back(basePtr);
  callArgTypes.push_back(plainBasePtrTy);

  callOperands.push_back(meta.getOffset());
  callArgTypes.push_back(indexType);

  for (Value stride : meta.getStrides()) {
    callOperands.push_back(stride);
    callArgTypes.push_back(indexType);
  }
}

//===----------------------------------------------------------------------===//
// External model — BufferizableOpInterface for mips.matmul.
//
// Inherits from DstBufferizableOpInterfaceExternalModel which automatically
// handles the DPS aliasing (init ↔ result) and write detection for the init
// operand.  We override bufferizesToMemoryRead to mark lhs and rhs as read,
// and provide a custom bufferize() that selects the right kernel based on the
// LHS element type.
//===----------------------------------------------------------------------===//

struct MIPSMatmulBufferizableOpInterface
    : public DstBufferizableOpInterfaceExternalModel<
          MIPSMatmulBufferizableOpInterface, MatmulOp> {

  // All three operands are read by the kernel.
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto matmulOp = cast<MatmulOp>(op);
    Location loc = matmulOp.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(matmulOp);

    // Obtain memref buffers for all three tensor operands.
    FailureOr<Value> lhsBuf =
        getBuffer(rewriter, matmulOp.getLhs(), options, state);
    if (failed(lhsBuf))
      return failure();
    FailureOr<Value> rhsBuf =
        getBuffer(rewriter, matmulOp.getRhs(), options, state);
    if (failed(rhsBuf))
      return failure();
    // init aliases with result — one-shot bufferize allocates the output buffer
    // (via bufferization.alloc_tensor or in-place analysis) and gives it here.
    FailureOr<Value> initBuf =
        getBuffer(rewriter, matmulOp.getInit(), options, state);
    if (failed(initBuf))
      return failure();

    // Select the kernel based on LHS element type.
    Type lhsElemTy = cast<MemRefType>(lhsBuf->getType()).getElementType();
    StringRef kernelName;
    if (lhsElemTy.isF32()) {
      kernelName = kKernelF32;
    } else if (lhsElemTy.isInteger(8)) {
      kernelName = kKernelI8;
    } else {
      return matmulOp.emitOpError(
          "MIPSBufferizableOpInterface: unsupported LHS element type '")
             << lhsElemTy
             << "'; supported types are f32 and i8";
    }

    // Build the flattened argument list for the kernel call.
    //   For each 2-D memref: (base_ptr, offset, stride0, stride1)
    //   Then: M, N, K as index scalars.
    SmallVector<Value> callOperands;
    SmallVector<Type> callArgTypes;

    decomposeMemref2D(rewriter, loc, *lhsBuf, callOperands, callArgTypes);
    decomposeMemref2D(rewriter, loc, *rhsBuf, callOperands, callArgTypes);
    decomposeMemref2D(rewriter, loc, *initBuf, callOperands, callArgTypes);

    Type indexType = IndexType::get(ctx);
    Value M = memref::DimOp::create(rewriter, loc, *lhsBuf, 0);
    Value N = memref::DimOp::create(rewriter, loc, *rhsBuf, 1);
    Value K = memref::DimOp::create(rewriter, loc, *lhsBuf, 1);
    callOperands.append({M, N, K});
    callArgTypes.append(3, indexType);

    // Declare the kernel function in the enclosing module (idempotent).
    Operation *moduleOp = SymbolTable::getNearestSymbolTable(matmulOp);
    FunctionType fnType = rewriter.getFunctionType(callArgTypes, TypeRange{});
    ensureKernelDeclaration(rewriter, moduleOp, kernelName, fnType, loc);

    // Emit the call — the kernel writes into *initBuf in place.
    func::CallOp::create(rewriter, loc, kernelName, TypeRange{}, callOperands);

    // Replace the tensor result with the init buffer (DPS aliasing).
    replaceOpWithBufferizedValues(rewriter, op, *initBuf);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public registration entry point
//===----------------------------------------------------------------------===//

void registerMIPSBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, MIPSDialect * /*dialect*/) {
    MatmulOp::attachInterface<MIPSMatmulBufferizableOpInterface>(*ctx);
  });
}

} // namespace mlir::iree_compiler::IREE::MIPS
