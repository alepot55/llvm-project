// RUN: mlir-opt -split-input-file -gpu-annotate-uniform-collectives %s | FileCheck %s

// A reduction in control flow that no thread-dependent value steers is executed
// by the whole subgroup.
// CHECK-LABEL: gpu.func @straight_line
gpu.module @straight_line_module {
  gpu.func @straight_line(%n: index) kernel {
    %tid = gpu.thread_id x
    %v = arith.index_cast %tid : index to i32
    // CHECK: gpu.subgroup_reduce add {{.*}} uniform
    %r = gpu.subgroup_reduce add %v : (i32) -> (i32)
    // CHECK: gpu.all_reduce add {{.*}} uniform
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
      // CHECK: gpu.subgroup_reduce add
      // CHECK-NOT: uniform
      %r = gpu.subgroup_reduce add %v : (i32) -> (i32)
    }
    gpu.return
  }
}

// -----

// The block index is the same for every thread of the workgroup, so a condition
// on it narrows nothing.
// CHECK-LABEL: gpu.func @uniform_condition
gpu.module @uniform_condition_module {
  gpu.func @uniform_condition(%n: index) kernel {
    %bid = gpu.block_id x
    %tid = gpu.thread_id x
    %v = arith.index_cast %tid : index to i32
    %cond = arith.cmpi ult, %bid, %n : index
    scf.if %cond {
      // CHECK: gpu.subgroup_reduce add {{.*}} uniform
      %r = gpu.subgroup_reduce add %v : (i32) -> (i32)
    }
    gpu.return
  }
}

// -----

// A workgroup reduction under a subgroup-uniform condition: the subgroup
// reduction may be annotated, the workgroup one may not.
// CHECK-LABEL: gpu.func @subgroup_only
gpu.module @subgroup_only_module {
  gpu.func @subgroup_only(%n: index) kernel {
    %sg = gpu.subgroup_id : index
    %tid = gpu.thread_id x
    %v = arith.index_cast %tid : index to i32
    %cond = arith.cmpi ult, %sg, %n : index
    scf.if %cond {
      // CHECK: gpu.subgroup_reduce add {{.*}} uniform
      %r = gpu.subgroup_reduce add %v : (i32) -> (i32)
      // CHECK: gpu.all_reduce add
      // CHECK-NOT: uniform
      %s = gpu.all_reduce add %v {} : (i32) -> (i32)
    }
    gpu.return
  }
}
