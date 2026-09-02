//===- InferUniformityOpInterfaceImpls.cpp - GPU uniformity ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/InferUniformityOpInterface.h"

using namespace mlir;
using namespace mlir::gpu;

//===----------------------------------------------------------------------===//
// Thread identity and launch configuration
//===----------------------------------------------------------------------===//

/// The launch configuration is the same for every thread of the launch.
#define UNIFORM_OP(OpName)                                                     \
  void OpName::inferUniformity(ArrayRef<Uniformity>,                           \
                               SetUniformityFn setUniformity) {                \
    setUniformity(getResult(), UniformityScope::Uniform);                      \
  }

UNIFORM_OP(GridDimOp)
UNIFORM_OP(BlockDimOp)
UNIFORM_OP(ClusterDimOp)
UNIFORM_OP(ClusterDimBlocksOp)
UNIFORM_OP(NumSubgroupsOp)
UNIFORM_OP(SubgroupSizeOp)

#undef UNIFORM_OP

/// An index of the current thread differs between threads of a subgroup.
void ThreadIdOp::inferUniformity(ArrayRef<Uniformity>,
                                 SetUniformityFn setUniformity) {
  setUniformity(getResult(), UniformityScope::Divergent);
}

void GlobalIdOp::inferUniformity(ArrayRef<Uniformity>,
                                 SetUniformityFn setUniformity) {
  setUniformity(getResult(), UniformityScope::Divergent);
}

void LaneIdOp::inferUniformity(ArrayRef<Uniformity>,
                               SetUniformityFn setUniformity) {
  setUniformity(getResult(), UniformityScope::Divergent);
}

/// The index of the subgroup is the same for every lane of the subgroup.
void SubgroupIdOp::inferUniformity(ArrayRef<Uniformity>,
                                   SetUniformityFn setUniformity) {
  setUniformity(getResult(), UniformityScope::Subgroup);
}

/// The index of the workgroup is the same for every thread of the workgroup,
/// whether within the grid or within the cluster.
void BlockIdOp::inferUniformity(ArrayRef<Uniformity>,
                                SetUniformityFn setUniformity) {
  setUniformity(getResult(), UniformityScope::Workgroup);
}

void ClusterBlockIdOp::inferUniformity(ArrayRef<Uniformity>,
                                       SetUniformityFn setUniformity) {
  setUniformity(getResult(), UniformityScope::Workgroup);
}

void ClusterIdOp::inferUniformity(ArrayRef<Uniformity>,
                                  SetUniformityFn setUniformity) {
  setUniformity(getResult(), UniformityScope::Cluster);
}

//===----------------------------------------------------------------------===//
// gpu.launch / gpu.func
//===----------------------------------------------------------------------===//

static void setDim3(SetUniformityFn setUniformity, KernelDim3 dims,
                    UniformityScope scope) {
  setUniformity(dims.x, scope);
  setUniformity(dims.y, scope);
  setUniformity(dims.z, scope);
}

/// The body arguments of a launch: sizes are uniform, block indices are the
/// same within a workgroup, cluster indices within a cluster, thread indices
/// differ between threads.
void LaunchOp::inferUniformity(ArrayRef<Uniformity>,
                               SetUniformityFn setUniformity) {
  setDim3(setUniformity, getGridSize(), UniformityScope::Uniform);
  setDim3(setUniformity, getBlockSize(), UniformityScope::Uniform);
  setDim3(setUniformity, getBlockIds(), UniformityScope::Workgroup);
  setDim3(setUniformity, getThreadIds(), UniformityScope::Divergent);
  if (std::optional<KernelDim3> clusterSize = getClusterSize())
    setDim3(setUniformity, *clusterSize, UniformityScope::Uniform);
  if (std::optional<KernelDim3> clusterIds = getClusterIds())
    setDim3(setUniformity, *clusterIds, UniformityScope::Cluster);
}

/// Every thread of a launch receives the same kernel arguments. A workgroup
/// attribution is the same buffer for every thread of the workgroup; a private
/// attribution is a different buffer for every thread.
void GPUFuncOp::inferUniformity(ArrayRef<Uniformity>,
                                SetUniformityFn setUniformity) {
  if (!isKernel())
    return;
  for (BlockArgument argument :
       getBody().getArguments().take_front(getFunctionType().getNumInputs()))
    setUniformity(argument, UniformityScope::Uniform);
  for (BlockArgument attribution : getWorkgroupAttributionBBArgs())
    setUniformity(attribution, UniformityScope::Workgroup);
  for (BlockArgument attribution : getPrivateAttributions())
    setUniformity(attribution, UniformityScope::Divergent);
}

bool LaunchOp::isLaunchBoundary() { return true; }

//===----------------------------------------------------------------------===//
// Collectives
//===----------------------------------------------------------------------===//

/// Every thread of the workgroup receives the reduced value.
void AllReduceOp::inferUniformity(ArrayRef<Uniformity>,
                                  SetUniformityFn setUniformity) {
  setUniformity(getResult(), UniformityScope::Workgroup);
}

/// Every lane of the subgroup receives the reduced value, unless the subgroup
/// is split into clusters that each reduce separately.
void SubgroupReduceOp::inferUniformity(ArrayRef<Uniformity>,
                                       SetUniformityFn setUniformity) {
  setUniformity(getResult(), getClusterSize() ? UniformityScope::Divergent
                                              : UniformityScope::Subgroup);
}

/// A shuffle permutes values across lanes, so a uniform value stays uniform
/// and a divergent value generally does not become uniform; whether a lane
/// received a value depends on the lane. The width must be the same on every
/// lane, but that is a precondition, not something a shuffle establishes.
void ShuffleOp::inferUniformity(ArrayRef<Uniformity> operandUniformity,
                                SetUniformityFn setUniformity) {
  Uniformity value = operandUniformity[0];
  if (!value.isUninitialized())
    setUniformity(getShuffleResult(), value.getScope());
  setUniformity(getValid(), UniformityScope::Divergent);
}

void RotateOp::inferUniformity(ArrayRef<Uniformity> operandUniformity,
                               SetUniformityFn setUniformity) {
  Uniformity value = operandUniformity[0];
  if (!value.isUninitialized())
    setUniformity(getRotateResult(), value.getScope());
  setUniformity(getValid(), UniformityScope::Divergent);
}

/// A broadcast is the operation that makes a value uniform across the
/// subgroup.
void SubgroupBroadcastOp::inferUniformity(ArrayRef<Uniformity>,
                                          SetUniformityFn setUniformity) {
  setUniformity(getResult(), UniformityScope::Subgroup);
}

/// Every lane of the subgroup receives the same ballot mask.
void BallotOp::inferUniformity(ArrayRef<Uniformity>,
                               SetUniformityFn setUniformity) {
  setUniformity(getResult(), UniformityScope::Subgroup);
}

//===----------------------------------------------------------------------===//
// gpu.warp_execute_on_lane_0
//===----------------------------------------------------------------------===//

/// The region executes on a single lane, so within it every value is trivially
/// the same for all the threads executing it. A result whose type equals the
/// type yielded from the region is broadcast to every lane of the subgroup; a
/// result of a different type is distributed across the lanes.
void WarpExecuteOnLane0Op::inferUniformity(ArrayRef<Uniformity>,
                                           SetUniformityFn setUniformity) {
  for (BlockArgument argument : getWarpRegion().getArguments())
    setUniformity(argument, UniformityScope::Uniform);

  Operation *terminator = getBody()->getTerminator();
  for (auto [result, yielded] :
       llvm::zip_equal(getResults(), terminator->getOperands()))
    setUniformity(result, result.getType() == yielded.getType()
                              ? UniformityScope::Subgroup
                              : UniformityScope::Divergent);
}
