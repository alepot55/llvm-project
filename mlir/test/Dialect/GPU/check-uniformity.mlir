// RUN: mlir-opt -split-input-file -gpu-check-uniformity="warn-captured-values=true" -verify-diagnostics %s

// A barrier in control flow that depends on the thread index: the deadlock
// every GPU programmer has written once.
gpu.module @barrier_in_divergent_if {
  gpu.func @kernel(%n: index) kernel {
    %tid = gpu.thread_id x
    // expected-note @below {{the control flow depends on this value, which is divergent}}
    %cond = arith.cmpi ult, %tid, %n : index
    scf.if %cond {
      // expected-error @below {{'gpu.barrier' is executed in control flow that diverges within the workgroup}}
      gpu.barrier
    }
    gpu.return
  }
}

// -----

// The same barrier under a condition that is the same for every thread of the
// workgroup is fine.
gpu.module @barrier_in_uniform_if {
  gpu.func @kernel(%n: index) kernel {
    %bid = gpu.block_id x
    %cond = arith.cmpi ult, %bid, %n : index
    scf.if %cond {
      gpu.barrier
    }
    gpu.return
  }
}

// -----

// A subgroup barrier only needs the subgroup to agree; a workgroup barrier
// under the same condition does not.
gpu.module @scopes {
  gpu.func @kernel(%n: index) kernel {
    %sg = gpu.subgroup_id : index
    // expected-note @below {{the control flow depends on this value, which is subgroup}}
    %cond = arith.cmpi ult, %sg, %n : index
    scf.if %cond {
      gpu.barrier scope<subgroup>
      // expected-error @below {{'gpu.barrier' is executed in control flow that diverges within the workgroup}}
      gpu.barrier
    }
    gpu.return
  }
}

// -----

// A loop whose trip count depends on the thread index diverges too.
gpu.module @barrier_in_divergent_loop {
  gpu.func @kernel(%n: index, %m: memref<?xi32>) kernel {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %one = arith.constant 1 : i32
    // expected-note @below {{the control flow depends on this value, which is divergent}}
    %tid = gpu.thread_id x
    scf.for %i = %c0 to %tid step %c1 {
      // expected-error @below {{'gpu.all_reduce' is executed in control flow that diverges within the workgroup}}
      %r = gpu.all_reduce add %one {} : (i32) -> (i32)
      memref.store %r, %m[%i] : memref<?xi32>
    }
    gpu.return
  }
}

// -----

// The reduction is fine once the loop bounds come from the kernel arguments.
gpu.module @reduce_in_uniform_loop {
  gpu.func @kernel(%n: index, %m: memref<?xi32>) kernel {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      %v = arith.index_cast %i : index to i32
      %r = gpu.all_reduce add %v {} : (i32) -> (i32)
      memref.store %r, %m[%i] : memref<?xi32>
    }
    gpu.return
  }
}

// -----

// A subgroup reduction may run on a subset of lanes unless it is marked
// uniform.
gpu.module @subgroup_reduce {
  gpu.func @kernel(%n: index) kernel {
    %tid = gpu.thread_id x
    // expected-note @below {{the control flow depends on this value, which is divergent}}
    %cond = arith.cmpi ult, %tid, %n : index
    %v = arith.index_cast %tid : index to i32
    scf.if %cond {
      %ok = gpu.subgroup_reduce add %v : (i32) -> (i32)
      // expected-error @below {{'gpu.subgroup_reduce' marked uniform is executed in control flow that diverges within the subgroup}}
      %bad = gpu.subgroup_reduce add %v uniform : (i32) -> (i32)
    }
    gpu.return
  }
}

// -----

// Operands that must be the same across the subgroup.
gpu.module @operands {
  gpu.func @kernel(%n: index) kernel {
    %tid = gpu.thread_id x
    %v = arith.index_cast %tid : index to i32
    %c1 = arith.constant 1 : i32
    %c32 = arith.constant 32 : i32
    %ok, %valid = gpu.shuffle xor %v, %c1, %c32 : i32
    // expected-error @below {{the width of 'gpu.shuffle' must be uniform across the subgroup but is divergent}}
    %bad, %valid2 = gpu.shuffle xor %v, %c1, %v : i32
    %first = gpu.subgroup_broadcast %v, first_active_lane : i32
    %lane = gpu.subgroup_broadcast %v, first_active_lane : i32
    %fine = gpu.subgroup_broadcast %v, specific_lane %lane : i32
    // expected-error @below {{the lane of 'gpu.subgroup_broadcast' must be uniform across the subgroup but is divergent}}
    %poison = gpu.subgroup_broadcast %v, specific_lane %v : i32
    gpu.return
  }
}

// -----

// A shuffle needs the first `width` lanes of its subgroup to be active, which
// a condition that is not the same across the subgroup does not give it.
gpu.module @shuffle_in_divergent_if {
  gpu.func @kernel(%n: index) kernel {
    %tid = gpu.thread_id x
    %v = arith.index_cast %tid : index to i32
    %c1 = arith.constant 1 : i32
    %c32 = arith.constant 32 : i32
    // expected-note @below {{the control flow depends on this value, which is divergent}}
    %cond = arith.cmpi ult, %tid, %n : index
    scf.if %cond {
      // expected-error @below {{'gpu.shuffle' is executed in control flow that diverges within the subgroup}}
      %bad, %valid = gpu.shuffle xor %v, %c1, %c32 : i32
    }
    gpu.return
  }
}

// -----

// A rotate carries the same requirement; its width is an attribute, so only
// the execution can be wrong.
gpu.module @rotate_in_divergent_if {
  gpu.func @kernel(%n: index) kernel {
    %tid = gpu.thread_id x
    %v = arith.index_cast %tid : index to i32
    // expected-note @below {{the control flow depends on this value, which is divergent}}
    %cond = arith.cmpi ult, %tid, %n : index
    scf.if %cond {
      // expected-error @below {{'gpu.rotate' is executed in control flow that diverges within the subgroup}}
      %bad, %valid = gpu.rotate %v, 1, 32 : i32
    }
    gpu.return
  }
}

// -----

// Under a condition that is the same for every lane of the subgroup the whole
// subgroup reaches the shuffle, so there is nothing to report.
gpu.module @shuffle_in_uniform_if {
  gpu.func @kernel(%n: index) kernel {
    %tid = gpu.thread_id x
    %sg = gpu.subgroup_id : index
    %v = arith.index_cast %tid : index to i32
    %c1 = arith.constant 1 : i32
    %c32 = arith.constant 32 : i32
    %cond = arith.cmpi ult, %sg, %n : index
    scf.if %cond {
      %ok, %valid = gpu.shuffle xor %v, %c1, %c32 : i32
    }
    gpu.return
  }
}

// -----

// A launch under host control flow: the host condition does not narrow the
// execution of the body.
func.func @launch_under_host_if(%n: index, %go: i1) {
  %c1 = arith.constant 1 : index
  scf.if %go {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %n, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %n, %sy = %c1, %sz = %c1) {
      gpu.barrier
      gpu.terminator
    }
  }
  return
}

// -----

// Warp regions: every lane must reach the operation, and a captured value is
// only observed through lane 0.
func.func @warp(%n: index, %v: vector<32xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %n, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %n, %sy = %c1, %sz = %c1) {
    %laneid = gpu.lane_id
    // expected-warning @below {{'gpu.warp_execute_on_lane_0' captures a value that is divergent: only lane 0's copy is observed}}
    %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (index) {
      %s = arith.addi %tx, %c1 : index
      gpu.yield %s : index
    }
    // expected-note @below {{the control flow depends on this value, which is divergent}}
    %cond = arith.cmpi ult, %tx, %n : index
    scf.if %cond {
      // expected-error @below {{'gpu.warp_execute_on_lane_0' is executed in control flow that diverges within the subgroup}}
      gpu.warp_execute_on_lane_0(%laneid)[32] {
        gpu.yield
      }
    }
    gpu.terminator
  }
  return
}

// -----

// The interprocedural limit, pinned so that it is a decision and not a
// surprise: the barrier below is reached only by the threads that took the
// branch, but execution uniformity stops at the callable, so nothing is
// reported.
gpu.module @across_a_call {
  func.func @helper() {
    gpu.barrier
    return
  }
  gpu.func @kernel(%n: index) kernel {
    %tid = gpu.thread_id x
    %cond = arith.cmpi ult, %tid, %n : index
    scf.if %cond {
      func.call @helper() : () -> ()
    }
    gpu.return
  }
}
