//===- TestUniformityInstrumentation.cpp - Runtime uniformity oracle ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Instruments a GPU kernel so that a run on real hardware reports what every
// thread actually observed for every integer value the uniformity analysis
// reasoned about. Each instrumented value prints one line per executing
// thread:
//
//   UNI <valueId> <claimedScope> <blockLin> <threadLin> <occurrence> <value>
//
// `occurrence` counts, per thread, how many times that value has been defined
// so far, so that the k-th dynamic instance of a value in a loop is compared
// against the k-th instance in the other threads. A host-side checker groups
// the lines by (valueId, occurrence, group) and reports the claim unsound if
// two threads of the same group observed different values.
//
// The pass also writes, on stderr, one `UNIMAP <id> <scope> <location>` line
// per instrumented value, so that a failure can be traced back to the IR.
//
// This pass exists to validate the analysis against hardware; it is not part
// of any production pipeline.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/UniformityAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::dataflow;

namespace {

/// The linear block and thread identifiers of the running thread.
struct LaunchIds {
  Value blockLin;
  Value threadLin;
};

/// Widens `v` to an i64, or returns null if it is not an integer or an index.
static Value toI64(OpBuilder &builder, Location loc, Value v) {
  Type type = v.getType();
  Type i64 = builder.getI64Type();
  if (isa<IndexType>(type))
    return arith::IndexCastUIOp::create(builder, loc, i64, v);
  auto intType = dyn_cast<IntegerType>(type);
  if (!intType || !intType.isSignless())
    return nullptr;
  if (intType.getWidth() == 64)
    return v;
  if (intType.getWidth() < 64)
    return arith::ExtUIOp::create(builder, loc, i64, v);
  return arith::TruncIOp::create(builder, loc, i64, v);
}

/// Builds `x + dimX * (y + dimY * z)`, the usual x-major linearization.
static Value linearize(OpBuilder &builder, Location loc, Value x, Value y,
                       Value z, Value dimX, Value dimY) {
  Type indexType = builder.getIndexType();
  auto flags =
      arith::IntegerOverflowFlags::nsw | arith::IntegerOverflowFlags::nuw;
  Value dimYxZ = arith::MulIOp::create(builder, loc, indexType, dimY, z, flags);
  Value inner = arith::AddIOp::create(builder, loc, indexType, dimYxZ, y, flags);
  Value scaled =
      arith::MulIOp::create(builder, loc, indexType, dimX, inner, flags);
  return arith::AddIOp::create(builder, loc, indexType, x, scaled, flags);
}

/// Materializes the linear block and thread identifiers at the current
/// insertion point of `builder`.
static LaunchIds buildIds(OpBuilder &builder, Location loc) {
  Value tx = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);
  Value ty = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::y);
  Value tz = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::z);
  Value bdx = gpu::BlockDimOp::create(builder, loc, gpu::Dimension::x);
  Value bdy = gpu::BlockDimOp::create(builder, loc, gpu::Dimension::y);
  Value bx = gpu::BlockIdOp::create(builder, loc, gpu::Dimension::x);
  Value by = gpu::BlockIdOp::create(builder, loc, gpu::Dimension::y);
  Value bz = gpu::BlockIdOp::create(builder, loc, gpu::Dimension::z);
  Value gdx = gpu::GridDimOp::create(builder, loc, gpu::Dimension::x);
  Value gdy = gpu::GridDimOp::create(builder, loc, gpu::Dimension::y);
  return {linearize(builder, loc, bx, by, bz, gdx, gdy),
          linearize(builder, loc, tx, ty, tz, bdx, bdy)};
}

struct TestUniformityInstrumentationPass
    : public PassWrapper<TestUniformityInstrumentationPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestUniformityInstrumentationPass)

  StringRef getArgument() const override {
    return "test-uniformity-instrumentation";
  }
  StringRef getDescription() const override {
    return "Instrument every integer value of a GPU kernel with a device-side "
           "print of what each thread observed, to validate the uniformity "
           "analysis against hardware";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, gpu::GPUDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    DataFlowSolver solver;
    loadBaselineAnalyses(solver);
    solver.load<UniformityAnalysis>();
    if (failed(solver.initializeAndRun(module)))
      return signalPassFailure();

    int64_t nextId = 0;
    std::string mapping;
    module.walk([&](gpu::GPUFuncOp func) {
      if (func.isKernel() && !func.isExternal())
        instrument(solver, func.getBody(), nextId, mapping);
    });
    llvm::errs() << mapping;
  }

  /// Instruments every integer value defined in `region`.
  void instrument(DataFlowSolver &solver, Region &region, int64_t &nextId,
                  std::string &mapping) {
    Location loc = region.getParentOp()->getLoc();
    Block &entry = region.front();

    // Collect first: the walk must not see the operations this pass creates.
    SmallVector<Value> targets;
    for (BlockArgument arg : entry.getArguments())
      targets.push_back(arg);
    region.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<gpu::WarpExecuteOnLane0Op>(op))
        return WalkResult::skip();
      for (Region &nested : op->getRegions())
        for (Block &block : nested)
          for (BlockArgument arg : block.getArguments())
            targets.push_back(arg);
      for (Value result : op->getResults())
        targets.push_back(result);
      return WalkResult::advance();
    });
    llvm::erase_if(targets, [](Value v) {
      return !isa<IndexType, IntegerType>(v.getType());
    });
    if (targets.empty())
      return;

    // The prelude: the identifiers, then one thread-private counter per
    // instrumented value.
    OpBuilder builder(region.getContext());
    builder.setInsertionPointToStart(&entry);
    LaunchIds ids = buildIds(builder, loc);
    auto counterType = MemRefType::get({}, builder.getI64Type());
    Value zero =
        arith::ConstantOp::create(builder, loc, builder.getI64IntegerAttr(0));
    SmallVector<Value> counters;
    for (size_t i = 0, e = targets.size(); i < e; ++i) {
      Value counter = memref::AllocaOp::create(builder, loc, counterType);
      memref::StoreOp::create(builder, loc, zero, counter, ValueRange{});
      counters.push_back(counter);
    }
    Operation *preludeEnd = &*std::prev(builder.getInsertionPoint());

    for (auto [value, counter] : llvm::zip_equal(targets, counters))
      emitProbe(solver, ids, counter, preludeEnd, value, nextId++, mapping);
  }

  /// Emits the per-thread counter update and the print for `value`.
  void emitProbe(DataFlowSolver &solver, LaunchIds ids, Value counter,
                 Operation *preludeEnd, Value value, int64_t id,
                 std::string &mapping) {
    Location loc = value.getLoc();
    OpBuilder builder(value.getContext());
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      Block *block = arg.getOwner();
      if (block == preludeEnd->getBlock())
        builder.setInsertionPointAfter(preludeEnd);
      else
        builder.setInsertionPointToStart(block);
    } else {
      builder.setInsertionPointAfter(value.getDefiningOp());
    }

    Value occurrence = memref::LoadOp::create(builder, loc, counter,
                                              ValueRange{});
    Value one =
        arith::ConstantOp::create(builder, loc, builder.getI64IntegerAttr(1));
    Value next = arith::AddIOp::create(builder, loc, occurrence, one);
    memref::StoreOp::create(builder, loc, next, counter, ValueRange{});

    UniformityScope scope = getUniformity(solver, value);
    Value idConst =
        arith::ConstantOp::create(builder, loc, builder.getI64IntegerAttr(id));
    Value scopeConst = arith::ConstantOp::create(
        builder, loc, builder.getI64IntegerAttr(static_cast<int64_t>(scope)));
    Value blockLin = toI64(builder, loc, ids.blockLin);
    Value threadLin = toI64(builder, loc, ids.threadLin);
    Value payload = toI64(builder, loc, value);
    gpu::PrintfOp::create(builder, loc, "UNI %lld %lld %lld %lld %lld %lld\n",
                          ValueRange{idConst, scopeConst, blockLin, threadLin,
                                     occurrence, payload});

    llvm::raw_string_ostream os(mapping);
    os << "UNIMAP " << id << ' ' << stringifyUniformityScope(scope) << ' '
       << value.getLoc() << '\n';
  }
};
} // end anonymous namespace

namespace mlir::test {
void registerTestUniformityInstrumentationPass() {
  PassRegistration<TestUniformityInstrumentationPass>();
}
} // end namespace mlir::test
