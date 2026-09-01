//===- InferUniformityOpInterfaceImpl.cpp - Uniformity models ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/InferUniformityOpInterfaceImpl.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Interfaces/InferUniformityOpInterface.h"

using namespace mlir;

namespace {

/// Returns true if `type` is a memref in thread-private memory.
static bool isThreadPrivate(Type type) {
  auto memRefType = dyn_cast<BaseMemRefType>(type);
  if (!memRefType)
    return false;
  auto space =
      dyn_cast_if_present<gpu::AddressSpaceAttr>(memRefType.getMemorySpace());
  return space && space.getValue() == gpu::AddressSpace::Private;
}

/// A load observes, within a group of threads, the same value when every
/// operand (the memory, the indices, a mask) is the same within the group and
/// the memory is not thread-private. Two threads reading the same address of a
/// memory they share read the same value in the absence of a data race, which
/// is the assumption LLVM's uniformity analysis makes too.
template <typename LoadOp>
struct LoadOpModel
    : public InferUniformityOpInterface::ExternalModel<LoadOpModel<LoadOp>,
                                                       LoadOp> {
  void inferUniformity(Operation *op, ArrayRef<Uniformity> operandUniformity,
                       SetUniformityFn setUniformity) const {
    if (isThreadPrivate(op->getOperand(0).getType())) {
      setUniformity(op->getResult(0), UniformityScope::Divergent);
      return;
    }
    Uniformity joined = Uniformity::join(operandUniformity);
    if (!joined.isUninitialized())
      setUniformity(op->getResult(0), joined.getScope());
  }
};

/// The arguments of a kernel are the same for every thread of the launch.
struct KernelFuncOpModel
    : public InferUniformityOpInterface::ExternalModel<KernelFuncOpModel,
                                                       func::FuncOp> {
  void inferUniformity(Operation *op, ArrayRef<Uniformity>,
                       SetUniformityFn setUniformity) const {
    auto funcOp = cast<func::FuncOp>(op);
    if (!funcOp->hasAttr(gpu::GPUDialect::getKernelFuncAttrName()) ||
        funcOp.isExternal())
      return;
    for (BlockArgument argument : funcOp.getArguments())
      setUniformity(argument, UniformityScope::Uniform);
  }
};

} // namespace

void mlir::gpu::registerInferUniformityOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, memref::MemRefDialect *) {
    memref::LoadOp::attachInterface<LoadOpModel<memref::LoadOp>>(*ctx);
  });
  registry.addExtension(+[](MLIRContext *ctx, affine::AffineDialect *) {
    affine::AffineLoadOp::attachInterface<LoadOpModel<affine::AffineLoadOp>>(
        *ctx);
  });
  registry.addExtension(+[](MLIRContext *ctx, vector::VectorDialect *) {
    vector::LoadOp::attachInterface<LoadOpModel<vector::LoadOp>>(*ctx);
    vector::MaskedLoadOp::attachInterface<LoadOpModel<vector::MaskedLoadOp>>(
        *ctx);
    vector::TransferReadOp::attachInterface<
        LoadOpModel<vector::TransferReadOp>>(*ctx);
  });
  registry.addExtension(+[](MLIRContext *ctx, func::FuncDialect *) {
    func::FuncOp::attachInterface<KernelFuncOpModel>(*ctx);
  });
}
