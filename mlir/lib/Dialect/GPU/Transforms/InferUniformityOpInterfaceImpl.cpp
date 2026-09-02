//===- InferUniformityOpInterfaceImpl.cpp - Uniformity models -------------===//
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

#include <algorithm>

using namespace mlir;

/// Returns the widest group of threads that can observe the same value when
/// reading `type` at the same address: thread-private memory is a different
/// memory for every thread, workgroup memory is a different memory in every
/// workgroup, and global and constant memory are shared by the whole launch.
/// A memory whose space is not a `gpu::AddressSpaceAttr` is not known to be
/// per-thread, so it is treated as shared.
static UniformityScope memoryCeiling(Type type) {
  auto memRefType = dyn_cast<BaseMemRefType>(type);
  if (!memRefType)
    return UniformityScope::Uniform;
  auto space =
      dyn_cast_if_present<gpu::AddressSpaceAttr>(memRefType.getMemorySpace());
  if (!space)
    return UniformityScope::Uniform;
  switch (space.getValue()) {
  case gpu::AddressSpace::Private:
    return UniformityScope::Divergent;
  case gpu::AddressSpace::Workgroup:
    return UniformityScope::Workgroup;
  case gpu::AddressSpace::Global:
  case gpu::AddressSpace::Constant:
    return UniformityScope::Uniform;
  }
  llvm_unreachable("unknown GPU address space");
}

namespace {

/// A load observes, within a group of threads, the same value when every
/// operand (the memory, the indices, a mask) is the same within the group and
/// the group shares the memory. Two threads reading the same address of a
/// memory they share read the same value in the absence of a data race, which
/// is the assumption LLVM's uniformity analysis makes too. Sharing is what the
/// address space says: a workgroup memory is shared by a workgroup and no
/// wider, whatever the address, and a thread-private memory by nobody.
template <typename LoadOp>
struct LoadOpModel
    : public InferUniformityOpInterface::ExternalModel<LoadOpModel<LoadOp>,
                                                       LoadOp> {
  void inferUniformity(Operation *op, ArrayRef<Uniformity> operandUniformity,
                       SetUniformityFn setUniformity) const {
    UniformityScope ceiling = memoryCeiling(op->getOperand(0).getType());
    if (ceiling == UniformityScope::Divergent) {
      setUniformity(op->getResult(0), UniformityScope::Divergent);
      return;
    }
    Uniformity joined = Uniformity::join(operandUniformity);
    if (!joined.isUninitialized())
      setUniformity(op->getResult(0), std::min(joined.getScope(), ceiling));
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
