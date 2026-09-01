//===- UniformityAnalysis.h - Uniformity (divergence) analysis --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the dataflow analysis that computes, for SIMT programs
// expressed with structured control flow, the uniformity of every SSA value:
// the widest group of threads within which all threads observe the same value.
// Operations participate in the analysis by implementing
// `InferUniformityOpInterface`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_DATAFLOW_UNIFORMITYANALYSIS_H
#define MLIR_ANALYSIS_DATAFLOW_UNIFORMITYANALYSIS_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Interfaces/InferUniformityOpInterface.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/StringSet.h"

namespace mlir::dataflow {

/// The uniformity lattice element of an SSA value.
class UniformityLattice : public Lattice<Uniformity> {
public:
  using Lattice::Lattice;
};

/// Uniformity analysis determines, for each SSA value, the widest group of
/// threads (subgroup, workgroup, cluster, the whole launch) within which every
/// thread observes the same value. A value that two threads of the same
/// subgroup may observe differently is divergent.
///
/// A value `v` is uniform within a group at a program point `p` if all threads
/// of the group that execute the dynamic instance of `p` together observe the
/// same `v`. The analysis only reports uniformity that holds regardless of
/// which threads are active, so its facts stay true when a value is hoisted
/// out of divergent control flow.
///
/// Transfer functions:
///
/// - An operation implementing `InferUniformityOpInterface` describes the
///   uniformity of the values it defines itself.
/// - A memory-effect-free operation of a *transparent* dialect computes a
///   function of its operands, so its results are the join of the operands
///   (uniform when it has none). The default set of transparent dialects is
///   the one of the core dialects that have no notion of a thread.
/// - Any other operation defines divergent values: an operation that reads
///   memory, and an operation of a dialect that may read thread identity
///   without saying so through the interface.
/// - The results of an operation with region control flow are the join of the
///   values forwarded to them by region control flow, joined with the
///   uniformity of the operands that steer that control flow (the condition
///   of an `scf.if`, the bounds of an `scf.for`, the condition of an
///   `scf.condition`): threads that take different paths reach the results
///   with different values, and a loop whose trip count varies across threads
///   exits with different values. The entry block arguments of the regions
///   are only the join of the forwarded values, since they are observed by
///   the threads that entered the region together.
/// - In a region with unstructured control flow, the arguments of every
///   non-entry block are joined with the uniformity of the operands that
///   steer every branch of the region.
///
/// The analysis assumes structured reconvergence: all threads of a group that
/// execute an operation with region control flow together reconverge after
/// it. This analysis depends on DeadCodeAnalysis and will be a silent no-op if
/// DeadCodeAnalysis is not loaded in the same solver context.
class UniformityAnalysis
    : public SparseForwardDataFlowAnalysis<UniformityLattice> {
public:
  /// Creates the analysis. `transparentDialects` lists the dialects whose
  /// memory-effect-free operations are known to compute a function of their
  /// operands only; it defaults to `getDefaultTransparentDialects()`.
  explicit UniformityAnalysis(DataFlowSolver &solver,
                              ArrayRef<StringRef> transparentDialects =
                                  getDefaultTransparentDialects());

  /// The dialects treated as transparent by default: `affine`, `arith`,
  /// `bufferization`, `builtin`, `cf`, `complex`, `func`, `index`, `linalg`,
  /// `math`, `memref`, `scf`, `tensor`, `ub` and `vector`.
  static ArrayRef<StringRef> getDefaultTransparentDialects();

  LogicalResult initialize(Operation *top) override;

  LogicalResult visit(ProgramPoint *point) override;

  /// At an entry point, a value is divergent.
  void setToEntryState(UniformityLattice *lattice) override;

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const UniformityLattice *> operands,
                               ArrayRef<UniformityLattice *> results) override;

  /// Infers block arguments and results that region control flow does not
  /// forward a value to (loop induction variables, the body arguments of
  /// `gpu.launch`) through `InferUniformityOpInterface`.
  void visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor &successor,
      ValueRange nonSuccessorInputs,
      ArrayRef<UniformityLattice *> nonSuccessorInputLattices) override;

protected:
  /// Lets a callable implementing `InferUniformityOpInterface` (a GPU kernel)
  /// describe the uniformity of its arguments.
  void visitCallableOperation(
      CallableOpInterface callable,
      ArrayRef<AbstractSparseLattice *> argLattices) override;

  /// Joins the forwarded values as usual, then adds the control dependence on
  /// the operands steering the region branch; or defers entirely to
  /// `InferUniformityOpInterface` when the operation implements it for the
  /// successor being visited.
  void
  visitRegionSuccessors(ProgramPoint *point, RegionBranchOpInterface branch,
                        RegionSuccessor successor,
                        ArrayRef<AbstractSparseLattice *> lattices) override;

private:
  /// Returns true if `op` belongs to a transparent dialect.
  bool isTransparent(Operation *op) const;

  /// Runs the interface of `op` at `point` and joins what it reports into the
  /// lattices of the `candidates` it names. Returns which candidates it set.
  llvm::SmallBitVector
  inferThroughInterface(InferUniformityOpInterface op, ProgramPoint *point,
                        ValueRange candidates,
                        ArrayRef<AbstractSparseLattice *> lattices);

  /// Puts the lattices that `set` does not mark in the entry state.
  void setUnsetToEntryStates(const llvm::SmallBitVector &set,
                             ArrayRef<AbstractSparseLattice *> lattices);

  /// Joins the uniformity of the operands steering `branch` and the
  /// terminators of its regions into `lattices`, the result lattices of
  /// `branch`.
  void joinControlDependence(ProgramPoint *point, Operation *branch,
                             ArrayRef<AbstractSparseLattice *> lattices);

  /// Joins the uniformity of the operands steering every branch in the region
  /// of `block` into the arguments of `block`, if it is not an entry block.
  void visitUnstructuredBlockArguments(Block *block);

  llvm::StringSet<> transparentDialects;
};

/// Collects into `controlOperands` the operands of `op` that steer control
/// flow rather than being forwarded to a successor: the operands of an
/// operation implementing `RegionBranchOpInterface` that are not entry
/// successor operands of any region successor (the condition of `scf.if`, the
/// bounds of `scf.for`), the operands of a `RegionBranchTerminatorOpInterface`
/// that are not successor operands of any successor (the condition of
/// `scf.condition`), and the operands of a `BranchOpInterface` that are not
/// forwarded to any successor (the condition of `cf.cond_br`). Collects nothing
/// for any other operation.
void getControlOperands(Operation *op, SmallVectorImpl<Value> &controlOperands);

/// Returns the uniformity scope of `value` after `solver` has run. A value the
/// analysis did not reach is reported as divergent.
UniformityScope getUniformity(DataFlowSolver &solver, Value value);

/// Returns the widest group of threads that execute `op` together: the meet of
/// the uniformity of the operands steering every enclosing region branch and
/// every branch of a region with unstructured control flow, up to the closest
/// enclosing callable. A `gpu.barrier` whose execution uniformity is narrower
/// than its scope may deadlock. If `narrowingOperand` is given, it receives
/// the first control operand that narrows the execution to the returned scope,
/// or null when nothing does.
UniformityScope getExecutionUniformity(DataFlowSolver &solver, Operation *op,
                                       Value *narrowingOperand = nullptr);

} // namespace mlir::dataflow

#endif // MLIR_ANALYSIS_DATAFLOW_UNIFORMITYANALYSIS_H
