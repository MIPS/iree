// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Converts torch matmul ops → mips.matmul.
//
// Patterns handled:
//   ConvertAtenMmToMIPSMatmul   — torch.aten.mm (f32 or i8 inputs)
//   ConvertAtenIntMmToMIPSMatmul — torch.aten._int_mm (i8 × i8 → i32)
//
// Both patterns run inside the Torch input-conversion pipeline, BEFORE
// createConvertTorchToLinalgPass(), so they intercept the ops first.
//
// Since torch ops carry ValueTensorType (torch's tensor type), each pattern:
//   1. Casts operands to builtin RankedTensorType via ToBuiltinTensorOp.
//   2. Creates a zero-initialised init tensor (Destination Passing Style).
//   3. Emits mips.matmul on builtin tensors.
//   4. Casts the result back to ValueTensorType via FromBuiltinTensorOp.
//
// The mips.matmul op is eliminated during One-Shot Bufferize:
// MIPSBufferizableOpInterface detects the LHS element type and calls either
// my_matmul_kernel (f32) or my_matmul_kernel_i8 (i8→i32).
//

#include "compiler/plugins/input/Torch/InputConversion/Passes.h"
#include "iree/compiler/Dialect/MIPS/IR/MIPSOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

namespace mlir::iree_compiler::TorchInput {

#define GEN_PASS_DEF_CONVERTTORCHTOMIPSPASS
#include "compiler/plugins/input/Torch/InputConversion/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helper: normalize signed/unsigned integer types to signless.
//
// arith.constant (and most MLIR arithmetic ops) require signless integers.
// Torch's dtype mapping produces signed types (e.g. si8, si32) which must be
// converted to their signless equivalents (i8, i32) before entering arith/
// linalg/tensor dialects.
//===----------------------------------------------------------------------===//

static Type toSignlessElemType(MLIRContext *ctx, Type ty) {
  if (auto intTy = dyn_cast<IntegerType>(ty))
    if (!intTy.isSignless())
      return IntegerType::get(ctx, intTy.getWidth());
  return ty;
}

static RankedTensorType toSignlessTensorType(RankedTensorType ty) {
  Type elem = toSignlessElemType(ty.getContext(), ty.getElementType());
  if (elem == ty.getElementType()) return ty;
  return RankedTensorType::get(ty.getShape(), elem, ty.getEncoding());
}

//===----------------------------------------------------------------------===//
// Helper: create a zero-filled tensor of a given shape and element type.
// Accepts (M, N) as dynamic Value dimensions.
//===----------------------------------------------------------------------===//

static Value createZeroTensor(PatternRewriter &rewriter, Location loc,
                               RankedTensorType ty, ValueRange dynSizes) {
  // Use signless element types — arith.constant rejects signed integers.
  RankedTensorType signlessTy = toSignlessTensorType(ty);
  Value empty = tensor::EmptyOp::create(rewriter, loc, signlessTy, dynSizes);
  Attribute zeroAttr = rewriter.getZeroAttr(signlessTy.getElementType());
  Value zero = arith::ConstantOp::create(rewriter, loc, cast<TypedAttr>(zeroAttr));
  return linalg::FillOp::create(rewriter, loc, zero, empty).result();
}

//===----------------------------------------------------------------------===//
// Helper: check whether a torch dtype is one we handle in mips.matmul.
// Returns true for f32 and si8/i8 inputs.
//===----------------------------------------------------------------------===//

static bool isSupportedMmDtype(Type torchDtype) {
  return torchDtype.isF32() || torchDtype.isSignedInteger(8) ||
         torchDtype.isInteger(8);
}

//===----------------------------------------------------------------------===//
// Pattern: torch.aten.mm → mips.matmul
//
// Handles f32 × f32 → f32  (original path)
//     and i8  × i8  → i8   (rare, but supported via same mips.matmul)
//===----------------------------------------------------------------------===//

struct ConvertAtenMmToMIPSMatmul
    : public OpRewritePattern<torch::Torch::AtenMmOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(torch::Torch::AtenMmOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // ----------------------------------------------------------------
    // 1. Verify that we have supported tensor types.
    // ----------------------------------------------------------------
    auto lhsTorchTy =
        dyn_cast<torch::Torch::ValueTensorType>(op.getSelf().getType());
    auto rhsTorchTy =
        dyn_cast<torch::Torch::ValueTensorType>(op.getMat2().getType());
    auto resultTorchTy =
        dyn_cast<torch::Torch::ValueTensorType>(op.getType());

    if (!lhsTorchTy || !rhsTorchTy || !resultTorchTy)
      return rewriter.notifyMatchFailure(op, "expected ValueTensorType");

    if (!isSupportedMmDtype(lhsTorchTy.getDtype()))
      return rewriter.notifyMatchFailure(op, "unsupported dtype (f32 or i8 only)");

    // ----------------------------------------------------------------
    // 2. Cast operands from torch ValueTensorType → builtin RankedTensorType.
    // ----------------------------------------------------------------
    auto lhsBuiltinTy = dyn_cast_or_null<RankedTensorType>(
        lhsTorchTy.toBuiltinTensor());
    auto rhsBuiltinTy = dyn_cast_or_null<RankedTensorType>(
        rhsTorchTy.toBuiltinTensor());
    auto resultBuiltinTy = dyn_cast_or_null<RankedTensorType>(
        resultTorchTy.toBuiltinTensor());

    if (!lhsBuiltinTy || !rhsBuiltinTy || !resultBuiltinTy ||
        lhsBuiltinTy.getRank() != 2 || rhsBuiltinTy.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "expected 2-D ranked tensors");

    // Normalize signed integer element types to signless (arith requires it).
    lhsBuiltinTy = toSignlessTensorType(lhsBuiltinTy);
    rhsBuiltinTy = toSignlessTensorType(rhsBuiltinTy);
    resultBuiltinTy = toSignlessTensorType(resultBuiltinTy);

    Value lhs = torch::TorchConversion::ToBuiltinTensorOp::create(
        rewriter, loc, lhsBuiltinTy, op.getSelf());
    Value rhs = torch::TorchConversion::ToBuiltinTensorOp::create(
        rewriter, loc, rhsBuiltinTy, op.getMat2());

    // ----------------------------------------------------------------
    // 3. Collect dynamic dimension values for the result tensor (M, N).
    // ----------------------------------------------------------------
    SmallVector<Value> dynSizes;
    if (resultBuiltinTy.isDynamicDim(0))
      dynSizes.push_back(tensor::DimOp::create(rewriter, loc, lhs, 0));
    if (resultBuiltinTy.isDynamicDim(1))
      dynSizes.push_back(tensor::DimOp::create(rewriter, loc, rhs, 1));

    // ----------------------------------------------------------------
    // 4. Create a zero-initialised init tensor for DPS output.
    // ----------------------------------------------------------------
    Value init = createZeroTensor(rewriter, loc, resultBuiltinTy, dynSizes);

    // ----------------------------------------------------------------
    // 5. Emit mips.matmul on builtin tensors.
    // ----------------------------------------------------------------
    Value result =
        IREE::MIPS::MatmulOp::create(rewriter, loc, TypeRange{resultBuiltinTy},
                                     lhs, rhs, init)
            .getResult();

    // ----------------------------------------------------------------
    // 6. Cast result back to ValueTensorType so downstream torch passes can
    //    still operate on it until the type finalisation pass runs.
    // ----------------------------------------------------------------
    Value torchResult = torch::TorchConversion::FromBuiltinTensorOp::create(
        rewriter, loc, resultTorchTy, result);

    rewriter.replaceOp(op, torchResult);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: torch.aten._int_mm → mips.matmul
//
// torch.aten._int_mm: i8 × i8 → i32 (integer matrix multiply).
// This is the primary op produced by INT8 quantization pipelines (e.g.
// torch.ao.quantization, torchao).
//
// mips.matmul carries i8 LHS/RHS and i32 output; MIPSBufferizableOpInterface
// detects the i8 LHS element type and emits func.call @my_matmul_kernel_i8.
//===----------------------------------------------------------------------===//

struct ConvertAtenIntMmToMIPSMatmul
    : public OpRewritePattern<torch::Torch::Aten_IntMmOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(torch::Torch::Aten_IntMmOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // ----------------------------------------------------------------
    // 1. Verify operand and result types.
    // ----------------------------------------------------------------
    auto lhsTorchTy =
        dyn_cast<torch::Torch::ValueTensorType>(op.getSelf().getType());
    auto rhsTorchTy =
        dyn_cast<torch::Torch::ValueTensorType>(op.getMat2().getType());
    auto resultTorchTy =
        dyn_cast<torch::Torch::ValueTensorType>(op.getType());

    if (!lhsTorchTy || !rhsTorchTy || !resultTorchTy)
      return rewriter.notifyMatchFailure(op, "expected ValueTensorType");

    // _int_mm expects i8 inputs and i32 output.
    if (!lhsTorchTy.getDtype().isSignedInteger(8) &&
        !lhsTorchTy.getDtype().isInteger(8))
      return rewriter.notifyMatchFailure(op, "expected i8 lhs");

    // ----------------------------------------------------------------
    // 2. Cast to builtin tensor types (signless integers — arith requires it).
    // ----------------------------------------------------------------
    auto lhsBuiltinTy = dyn_cast_or_null<RankedTensorType>(
        lhsTorchTy.toBuiltinTensor());
    auto rhsBuiltinTy = dyn_cast_or_null<RankedTensorType>(
        rhsTorchTy.toBuiltinTensor());
    auto resultBuiltinTy = dyn_cast_or_null<RankedTensorType>(
        resultTorchTy.toBuiltinTensor());

    if (!lhsBuiltinTy || !rhsBuiltinTy || !resultBuiltinTy ||
        lhsBuiltinTy.getRank() != 2 || rhsBuiltinTy.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "expected 2-D ranked tensors");

    lhsBuiltinTy = toSignlessTensorType(lhsBuiltinTy);
    rhsBuiltinTy = toSignlessTensorType(rhsBuiltinTy);
    resultBuiltinTy = toSignlessTensorType(resultBuiltinTy);

    Value lhs = torch::TorchConversion::ToBuiltinTensorOp::create(
        rewriter, loc, lhsBuiltinTy, op.getSelf());
    Value rhs = torch::TorchConversion::ToBuiltinTensorOp::create(
        rewriter, loc, rhsBuiltinTy, op.getMat2());

    // ----------------------------------------------------------------
    // 3. Dynamic dims for the i32 result tensor (M from lhs, N from rhs).
    // ----------------------------------------------------------------
    SmallVector<Value> dynSizes;
    if (resultBuiltinTy.isDynamicDim(0))
      dynSizes.push_back(tensor::DimOp::create(rewriter, loc, lhs, 0));
    if (resultBuiltinTy.isDynamicDim(1))
      dynSizes.push_back(tensor::DimOp::create(rewriter, loc, rhs, 1));

    // ----------------------------------------------------------------
    // 4. Zero-initialised i32 init tensor.
    // ----------------------------------------------------------------
    Value init = createZeroTensor(rewriter, loc, resultBuiltinTy, dynSizes);

    // ----------------------------------------------------------------
    // 5. Emit mips.matmul — LHS is i8, result is i32.
    //    MIPSBufferizableOpInterface dispatches to my_matmul_kernel_i8.
    // ----------------------------------------------------------------
    Value result =
        IREE::MIPS::MatmulOp::create(rewriter, loc, TypeRange{resultBuiltinTy},
                                     lhs, rhs, init)
            .getResult();

    // ----------------------------------------------------------------
    // 6. Cast result back to torch ValueTensorType.
    // ----------------------------------------------------------------
    Value torchResult = torch::TorchConversion::FromBuiltinTensorOp::create(
        rewriter, loc, resultTorchTy, result);

    rewriter.replaceOp(op, torchResult);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct ConvertTorchToMIPSPass
    : impl::ConvertTorchToMIPSPassBase<ConvertTorchToMIPSPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::MIPS::MIPSDialect,
                    torch::TorchConversion::TorchConversionDialect,
                    arith::ArithDialect, tensor::TensorDialect,
                    linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertAtenMmToMIPSMatmul>(context);
    patterns.add<ConvertAtenIntMmToMIPSMatmul>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
} // namespace mlir::iree_compiler::TorchInput
