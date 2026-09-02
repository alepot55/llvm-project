//===- AnnotateUniformCollectives.cpp - Infer the uniform flag ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that runs uniformity analysis and sets the
// `uniform` flag on the collectives it proves are executed by either all or
// none of the threads of their group. The flag is what lets a target lower a
// reduction to its convergent instruction, and today it can only be written by
// hand, unchecked.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/UniformityAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/IR/Operation.h"

namespace mlir {
#define GEN_PASS_DEF_GPUANNOTATEUNIFORMCOLLECTIVESPASS
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dataflow;

namespace {

struct GpuAnnotateUniformCollectivesPass
    : public impl::GpuAnnotateUniformCollectivesPassBase<
          GpuAnnotateUniformCollectivesPass> {
  using GpuAnnotateUniformCollectivesPassBase::
      GpuAnnotateUniformCollectivesPassBase;

  void runOnOperation() override {
    DataFlowSolver solver;
    loadBaselineAnalyses(solver);
    solver.load<UniformityAnalysis>();
    if (failed(solver.initializeAndRun(getOperation())))
      return signalPassFailure();

    // `uniform` promises that either all threads of the group execute the
    // operation or none do, which is what it means for the widest group
    // executing it together to be at least the group the collective reduces
    // over.
    auto annotate = [&](Operation *op, UniformityScope required) {
      if (op->hasAttr("uniform"))
        return;
      if (getExecutionUniformity(solver, op) < required)
        return;
      op->setAttr("uniform", UnitAttr::get(op->getContext()));
    };

    getOperation()->walk([&](Operation *op) {
      if (isa<gpu::SubgroupReduceOp>(op))
        annotate(op, UniformityScope::Subgroup);
      else if (isa<gpu::AllReduceOp>(op))
        annotate(op, UniformityScope::Workgroup);
    });
  }
};
} // end anonymous namespace
