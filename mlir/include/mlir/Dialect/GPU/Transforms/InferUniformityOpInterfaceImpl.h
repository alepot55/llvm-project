//===- InferUniformityOpInterfaceImpl.h - Uniformity ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_TRANSFORMS_INFERUNIFORMITYOPINTERFACEIMPL_H
#define MLIR_DIALECT_GPU_TRANSFORMS_INFERUNIFORMITYOPINTERFACEIMPL_H

namespace mlir {

class DialectRegistry;

namespace gpu {
/// Registers the `InferUniformityOpInterface` models that need the GPU
/// execution model to be stated: a load from an address that is the same for a
/// group of threads observes the same value within that group unless the
/// memory is thread-private, and the arguments of a `func.func` marked
/// `gpu.kernel` are the same for every thread.
void registerInferUniformityOpInterfaceExternalModels(
    DialectRegistry &registry);
} // namespace gpu
} // namespace mlir

#endif // MLIR_DIALECT_GPU_TRANSFORMS_INFERUNIFORMITYOPINTERFACEIMPL_H
