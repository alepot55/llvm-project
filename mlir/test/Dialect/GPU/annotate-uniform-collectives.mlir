// RUN: mlir-opt -split-input-file -gpu-annotate-uniform-collectives %s | FileCheck %s

// A reduction that no thread-dependent value steers is executed by the whole
// subgroup, and by the whole workgroup.
// CHECK-LABEL: gpu.func @straight_line
gpu.module @straight_line_module {
  gpu.func @straight_line(%n: index) kernel {
    %tid = gpu.thread_id x
    %v = arith.index_cast %tid : index to i32
    // CHECK: gpu.subgroup_reduce add %{{[0-9]+}} uniform :
    %r = gpu.subgroup_reduce add %v : (i32) -> (i32)
    // CHECK: gpu.all_reduce add %{{[0-9]+}} uniform {
    %s = gpu.all_reduce add %v {} : (i32) -> (i32)
    gpu.return
  }
}

// -----

// A reduction under a condition that differs within the subgroup is left alone.
// CHECK-LABEL: gpu.func @divergent
gpu.module @divergent_module {
  gpu.func @divergent(%n: index) kernel {
    %tid = gpu.thread_id x
    %v = arith.index_cast %tid : index to i32
    %cond = arith.cmpi ult, %tid, %n : index
    scf.if %cond {
      // CHECK: gpu.subgroup_reduce add %{{[0-9]+}} :
      %r = gpu.subgroup_reduce add %v : (i32) -> (i32)
    }
    gpu.return
  }
}

// -----

// The block index is the same for every thread of the workgroup, so a condition
// on it narrows nothing.
// CHECK-LABEL: gpu.func @block_condition
gpu.module @block_condition_module {
  gpu.func @block_condition(%n: index) kernel {
    %bid = gpu.block_id x
    %tid = gpu.thread_id x
    %v = arith.index_cast %tid : index to i32
    %cond = arith.cmpi ult, %bid, %n : index
    scf.if %cond {
      // CHECK: gpu.subgroup_reduce add %{{[0-9]+}} uniform :
      %r = gpu.subgroup_reduce add %v : (i32) -> (i32)
    }
    gpu.return
  }
}

// -----

// Under a condition on the subgroup index the subgroup reduction may be
// annotated and the workgroup one may not.
// CHECK-LABEL: gpu.func @subgroup_only
gpu.module @subgroup_only_module {
  gpu.func @subgroup_only(%n: index) kernel {
    %sg = gpu.subgroup_id : index
    %tid = gpu.thread_id x
    %v = arith.index_cast %tid : index to i32
    %cond = arith.cmpi ult, %sg, %n : index
    scf.if %cond {
      // CHECK: gpu.subgroup_reduce add %{{[0-9]+}} uniform :
      %r = gpu.subgroup_reduce add %v : (i32) -> (i32)
      // CHECK: gpu.all_reduce add %{{[0-9]+}} {
      %s = gpu.all_reduce add %v {} : (i32) -> (i32)
    }
    gpu.return
  }
}

// -----

// A flag the source already carries is a fact the analysis may not be able to
// see, so the pass leaves it alone.
// CHECK-LABEL: gpu.func @already_flagged
gpu.module @already_flagged_module {
  gpu.func @already_flagged(%n: index) kernel {
    %tid = gpu.thread_id x
    %v = arith.index_cast %tid : index to i32
    %cond = arith.cmpi ult, %tid, %n : index
    scf.if %cond {
      // CHECK: gpu.subgroup_reduce add %{{[0-9]+}} uniform :
      %r = gpu.subgroup_reduce add %v uniform : (i32) -> (i32)
    }
    gpu.return
  }
}
