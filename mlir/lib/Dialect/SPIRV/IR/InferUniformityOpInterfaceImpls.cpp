//===- InferUniformityOpInterfaceImpls.cpp - SPIR-V uniformity ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the uniformity inference interface for the group and subgroup
// operations of the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Interfaces/InferUniformityOpInterface.h"

namespace mlir::spirv {

/// The group of invocations that a SPIR-V execution scope names. A device
/// scope covers the whole launch; the invocation scope, and the scopes that do
/// not name a group of invocations executing together, promise nothing.
static UniformityScope getUniformityScope(Scope scope) {
  switch (scope) {
  case Scope::Subgroup:
    return UniformityScope::Subgroup;
  case Scope::Workgroup:
    return UniformityScope::Workgroup;
  case Scope::Device:
  case Scope::CrossDevice:
    return UniformityScope::Uniform;
  case Scope::Invocation:
  case Scope::QueueFamily:
  case Scope::ShaderCallKHR:
    return UniformityScope::Divergent;
  }
  llvm_unreachable("unknown SPIR-V scope");
}

/// A reduction delivers the reduced value to every invocation of the scope. A
/// scan gives each invocation its own prefix, and a clustered or partitioned
/// operation reduces within a subset of the scope.
static UniformityScope getGroupOperationScope(Scope scope,
                                              GroupOperation operation) {
  return operation == GroupOperation::Reduce ? getUniformityScope(scope)
                                             : UniformityScope::Divergent;
}

//===----------------------------------------------------------------------===//
// Results that are the same for every invocation of the execution scope
//===----------------------------------------------------------------------===//

/// A ballot, a broadcast and a group-wide predicate deliver the same value to
/// every invocation of the scope.
#define SCOPE_OP(OpName)                                                       \
  void OpName::inferUniformity(ArrayRef<Uniformity>,                           \
                               SetUniformityFn setUniformity) {                \
    setUniformity(getResult(), getUniformityScope(getExecutionScope()));       \
  }

SCOPE_OP(GroupNonUniformBallotOp)
SCOPE_OP(GroupNonUniformBroadcastOp)
SCOPE_OP(GroupNonUniformBroadcastFirstOp)
SCOPE_OP(GroupNonUniformAllOp)
SCOPE_OP(GroupNonUniformAnyOp)
SCOPE_OP(GroupNonUniformAllEqualOp)
SCOPE_OP(GroupBroadcastOp)

#undef SCOPE_OP

/// Every invocation of the subgroup receives the same ballot mask.
void KHRSubgroupBallotOp::inferUniformity(ArrayRef<Uniformity>,
                                          SetUniformityFn setUniformity) {
  setUniformity(getResult(), UniformityScope::Subgroup);
}

/// Exactly one invocation of the scope is elected.
void GroupNonUniformElectOp::inferUniformity(ArrayRef<Uniformity>,
                                             SetUniformityFn setUniformity) {
  setUniformity(getResult(), UniformityScope::Divergent);
}

//===----------------------------------------------------------------------===//
// Reductions and scans
//===----------------------------------------------------------------------===//

/// The result of a reduction is the same for every invocation of the scope,
/// whatever the value reduced; the result of a scan is not, even for a value
/// that is the same on every invocation.
#define GROUP_OPERATION_OP(OpName)                                             \
  void OpName::inferUniformity(ArrayRef<Uniformity>,                           \
                               SetUniformityFn setUniformity) {                \
    setUniformity(getResult(), getGroupOperationScope(getExecutionScope(),     \
                                                      getGroupOperation()));   \
  }

GROUP_OPERATION_OP(GroupNonUniformFAddOp)
GROUP_OPERATION_OP(GroupNonUniformFMaxOp)
GROUP_OPERATION_OP(GroupNonUniformFMinOp)
GROUP_OPERATION_OP(GroupNonUniformFMulOp)
GROUP_OPERATION_OP(GroupNonUniformIAddOp)
GROUP_OPERATION_OP(GroupNonUniformIMulOp)
GROUP_OPERATION_OP(GroupNonUniformSMaxOp)
GROUP_OPERATION_OP(GroupNonUniformSMinOp)
GROUP_OPERATION_OP(GroupNonUniformUMaxOp)
GROUP_OPERATION_OP(GroupNonUniformUMinOp)
GROUP_OPERATION_OP(GroupNonUniformBitwiseAndOp)
GROUP_OPERATION_OP(GroupNonUniformBitwiseOrOp)
GROUP_OPERATION_OP(GroupNonUniformBitwiseXorOp)
GROUP_OPERATION_OP(GroupNonUniformLogicalAndOp)
GROUP_OPERATION_OP(GroupNonUniformLogicalOrOp)
GROUP_OPERATION_OP(GroupNonUniformLogicalXorOp)
GROUP_OPERATION_OP(GroupFAddOp)
GROUP_OPERATION_OP(GroupFMaxOp)
GROUP_OPERATION_OP(GroupFMinOp)
GROUP_OPERATION_OP(GroupIAddOp)
GROUP_OPERATION_OP(GroupSMaxOp)
GROUP_OPERATION_OP(GroupSMinOp)
GROUP_OPERATION_OP(GroupUMaxOp)
GROUP_OPERATION_OP(GroupUMinOp)
GROUP_OPERATION_OP(GroupFMulKHROp)
GROUP_OPERATION_OP(GroupIMulKHROp)

#undef GROUP_OPERATION_OP

/// The count of the bits set in a ballot is a function of the ballot; the
/// scans count the bits at or below the invocation's own position.
void GroupNonUniformBallotBitCountOp::inferUniformity(
    ArrayRef<Uniformity> operandUniformity, SetUniformityFn setUniformity) {
  if (getGroupOperation() != GroupOperation::Reduce) {
    setUniformity(getResult(), UniformityScope::Divergent);
    return;
  }
  Uniformity value = operandUniformity[0];
  if (!value.isUninitialized())
    setUniformity(getResult(), value.getScope());
}

//===----------------------------------------------------------------------===//
// Results that have the uniformity of the value operand
//===----------------------------------------------------------------------===//

/// A shuffle, a rotation or a quad swap permutes the copies of a value across
/// the invocations: a value that is the same within a group stays the same,
/// and a value that differs does not become the same; the index only picks
/// which copy. The position of a bit in a ballot is a function of the ballot.
/// Nothing is said while the uniformity of the value is not known yet.
#define SAME_AS_VALUE_OP(OpName)                                               \
  void OpName::inferUniformity(ArrayRef<Uniformity> operandUniformity,         \
                               SetUniformityFn setUniformity) {                \
    Uniformity value = operandUniformity[0];                                   \
    if (!value.isUninitialized())                                              \
      setUniformity(getResult(), value.getScope());                            \
  }

SAME_AS_VALUE_OP(GroupNonUniformShuffleOp)
SAME_AS_VALUE_OP(GroupNonUniformShuffleXorOp)
SAME_AS_VALUE_OP(GroupNonUniformShuffleUpOp)
SAME_AS_VALUE_OP(GroupNonUniformShuffleDownOp)
SAME_AS_VALUE_OP(GroupNonUniformRotateKHROp)
SAME_AS_VALUE_OP(GroupNonUniformQuadSwapOp)
SAME_AS_VALUE_OP(GroupNonUniformBallotFindLSBOp)
SAME_AS_VALUE_OP(GroupNonUniformBallotFindMSBOp)

#undef SAME_AS_VALUE_OP

} // namespace mlir::spirv
