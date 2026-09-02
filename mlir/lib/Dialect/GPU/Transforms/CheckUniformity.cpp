//===- CheckUniformity.cpp - Uniformity of collectives --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that runs uniformity analysis and reports the
// operations whose semantics require a group of threads to execute them
// together but that sit in control flow steered by a value that differs
// within that group, as well as operands that must be uniform and are not.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlow/UniformityAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_GPUCHECKUNIFORMITYPASS
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dataflow;

static UniformityScope toUniformityScope(gpu::BarrierScope scope) {
  switch (scope) {
  case gpu::BarrierScope::Subgroup:
    return UniformityScope::Subgroup;
  case gpu::BarrierScope::Workgroup:
    return UniformityScope::Workgroup;
  case gpu::BarrierScope::Cluster:
    return UniformityScope::Cluster;
  }
  llvm_unreachable("unknown barrier scope");
}

namespace {

struct GpuCheckUniformityPass
    : public impl::GpuCheckUniformityPassBase<GpuCheckUniformityPass> {
  using GpuCheckUniformityPassBase::GpuCheckUniformityPassBase;

  void runOnOperation() override {
    DataFlowSolver solver;
    loadBaselineAnalyses(solver);
    solver.load<UniformityAnalysis>();
    if (failed(solver.initializeAndRun(getOperation())))
      return signalPassFailure();

    bool reportedError = false;

    // Reports `op` if the threads of `required` do not all execute it.
    auto checkExecution = [&](Operation *op, UniformityScope required,
                              StringRef what) {
      Value narrowing;
      UniformityScope actual = getExecutionUniformity(solver, op, &narrowing);
      if (actual >= required)
        return;
      InFlightDiagnostic diag = op->emitError()
                                << what << " is executed in control flow that "
                                << "diverges within the "
                                << stringifyUniformityScope(required);
      if (narrowing)
        diag.attachNote(narrowing.getLoc())
            << "the control flow depends on this value, which is "
            << stringifyUniformityScope(getUniformity(solver, narrowing));
      reportedError = true;
    };

    // Reports `operand` of `op` if it is not the same across the subgroup.
    auto checkSubgroupUniformOperand = [&](Operation *op, Value operand,
                                           StringRef what) {
      UniformityScope actual = getUniformity(solver, operand);
      if (actual >= UniformityScope::Subgroup)
        return;
      op->emitError() << what << " must be uniform across the subgroup but is "
                      << stringifyUniformityScope(actual);
      reportedError = true;
    };

    getOperation()->walk([&](Operation *op) {
      llvm::TypeSwitch<Operation *>(op)
          .Case([&](gpu::BarrierOp barrier) {
            checkExecution(barrier, toUniformityScope(barrier.getScope()),
                           "'gpu.barrier'");
          })
          .Case([&](gpu::AllReduceOp reduce) {
            checkExecution(reduce, UniformityScope::Workgroup,
                           "'gpu.all_reduce'");
          })
          .Case([&](gpu::SubgroupReduceOp reduce) {
            if (reduce.getUniform())
              checkExecution(reduce, UniformityScope::Subgroup,
                             "'gpu.subgroup_reduce' marked uniform");
          })
          .Case([&](gpu::ShuffleOp shuffle) {
            checkSubgroupUniformOperand(shuffle, shuffle.getWidth(),
                                        "the width of 'gpu.shuffle'");
          })
          .Case([&](gpu::SubgroupBroadcastOp broadcast) {
            if (Value lane = broadcast.getLane())
              checkSubgroupUniformOperand(
                  broadcast, lane, "the lane of 'gpu.subgroup_broadcast'");
          })
          .Case([&](gpu::WarpExecuteOnLane0Op warpOp) {
            checkExecution(warpOp, UniformityScope::Subgroup,
                           "'gpu.warp_execute_on_lane_0'");
            if (!warnCapturedValues)
              return;
            // Only lane 0's copy of a captured value is observable inside the
            // region, which is fine only when every lane holds the same.
            SetVector<Value> captured;
            getUsedValuesDefinedAbove(warpOp.getWarpRegion(), captured);
            for (Value value : captured) {
              UniformityScope actual = getUniformity(solver, value);
              if (actual >= UniformityScope::Subgroup)
                continue;
              warpOp.emitWarning()
                  << "'gpu.warp_execute_on_lane_0' captures a value that is "
                  << stringifyUniformityScope(actual)
                  << ": only lane 0's copy is observed";
            }
          });
    });

    if (reportedError)
      signalPassFailure();
  }
};

} // namespace
