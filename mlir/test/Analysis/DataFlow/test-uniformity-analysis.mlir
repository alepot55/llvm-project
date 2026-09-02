// RUN: mlir-opt -split-input-file -test-uniformity-analysis -verify-diagnostics %s

// Thread identity and launch configuration.
gpu.module @sources {
  gpu.func @kernel(%arg: index) kernel {
    // expected-remark @below {{uniformity of "tid": results = [divergent], execution = uniform}}
    %tid = gpu.thread_id x {tag = "tid"}
    // expected-remark @below {{uniformity of "lane": results = [divergent], execution = uniform}}
    %lane = gpu.lane_id {tag = "lane"}
    // expected-remark @below {{uniformity of "sg": results = [subgroup], execution = uniform}}
    %sg = gpu.subgroup_id {tag = "sg"} : index
    // expected-remark @below {{uniformity of "bid": results = [workgroup], execution = uniform}}
    %bid = gpu.block_id y {tag = "bid"}
    // expected-remark @below {{uniformity of "cid": results = [cluster], execution = uniform}}
    %cid = gpu.cluster_id x {tag = "cid"}
    // expected-remark @below {{uniformity of "bdim": results = [uniform], execution = uniform}}
    %bdim = gpu.block_dim x {tag = "bdim"}
    // expected-remark @below {{uniformity of "sgsize": results = [uniform], execution = uniform}}
    %sgsize = gpu.subgroup_size {tag = "sgsize"} : index
    // Kernel arguments are the same for every thread.
    // expected-remark @below {{uniformity of "arg_use": results = [uniform], execution = uniform}}
    %arg_use = arith.addi %arg, %arg {tag = "arg_use"} : index
    // Constants are uniform; arithmetic joins its operands.
    // expected-remark @below {{uniformity of "c": results = [uniform], execution = uniform}}
    %c = arith.constant {tag = "c"} 4 : index
    // expected-remark @below {{uniformity of "tid_c": results = [divergent], execution = uniform}}
    %tid_c = arith.muli %tid, %c {tag = "tid_c"} : index
    // expected-remark @below {{uniformity of "bid_c": results = [workgroup], execution = uniform}}
    %bid_c = arith.addi %bid, %c {tag = "bid_c"} : index
    // expected-remark @below {{uniformity of "bid_sg": results = [subgroup], execution = uniform}}
    %bid_sg = arith.addi %bid, %sg {tag = "bid_sg"} : index
    // expected-remark @below {{uniformity of "bid_cid": results = [workgroup], execution = uniform}}
    %bid_cid = arith.addi %bid, %cid {tag = "bid_cid"} : index
    gpu.return
  }
}

// -----

// Collectives.
gpu.module @collectives {
  gpu.func @kernel(%arg: f32) kernel {
    %tid = gpu.thread_id x
    %v = arith.index_cast %tid : index to i32
    // expected-remark @below {{uniformity of "all": results = [workgroup], execution = uniform}}
    %all = gpu.all_reduce add %v {} {tag = "all"} : (i32) -> (i32)
    // expected-remark @below {{uniformity of "sub": results = [subgroup], execution = uniform}}
    %sub = gpu.subgroup_reduce add %v {tag = "sub"} : (i32) -> (i32)
    // expected-remark @below {{uniformity of "clustered": results = [divergent], execution = uniform}}
    %clustered = gpu.subgroup_reduce add %v cluster(size = 4) {tag = "clustered"} : (i32) -> (i32)
    // expected-remark @below {{uniformity of "bcast": results = [subgroup], execution = uniform}}
    %bcast = gpu.subgroup_broadcast %v, first_active_lane {tag = "bcast"} : i32
    %pred = arith.cmpi ult, %v, %v : i32
    // expected-remark @below {{uniformity of "ballot": results = [subgroup], execution = uniform}}
    %ballot = gpu.ballot %pred {tag = "ballot"} : i32
    %w = arith.constant 32 : i32
    %o = arith.constant 1 : i32
    // A shuffle of a divergent value is divergent, of a uniform value uniform.
    // expected-remark @below {{uniformity of "shfl_div": results = [divergent, divergent], execution = uniform}}
    %s0, %valid0 = gpu.shuffle xor %v, %o, %w {tag = "shfl_div"} : i32
    // expected-remark @below {{uniformity of "shfl_uni": results = [uniform, divergent], execution = uniform}}
    %s1, %valid1 = gpu.shuffle xor %w, %o, %w {tag = "shfl_uni"} : i32
    gpu.return
  }
}

// -----

// Structured control flow: the results of a region branch are tainted by the
// operands steering it, the values inside are not.
gpu.module @scf {
  gpu.func @kernel(%n: index, %flag: i1) kernel {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tid = gpu.thread_id x
    %cond = arith.cmpi ult, %tid, %n : index
    // expected-remark @below {{uniformity of "if_div": results = [divergent], execution = uniform}}
    %r0 = scf.if %cond -> index {
      // expected-remark @below {{uniformity of "in_then": results = [uniform], execution = divergent}}
      %a = arith.addi %c1, %c1 {tag = "in_then"} : index
      scf.yield %a : index
    } else {
      scf.yield %c0 : index
    } {tag = "if_div"}
    // expected-remark @below {{uniformity of "if_uni": results = [uniform], execution = uniform}}
    %r1 = scf.if %flag -> index {
      scf.yield %c1 : index
    } else {
      scf.yield %c0 : index
    } {tag = "if_uni"}
    // A loop with uniform bounds has uniform induction variable and results.
    // expected-remark @below {{uniformity of "for_uni": results = [uniform], execution = uniform}}
    %r2 = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0) -> index {
      // expected-remark @below {{uniformity of "iv_uni": results = [uniform], execution = uniform}}
      %next = arith.addi %acc, %i {tag = "iv_uni"} : index
      scf.yield %next : index
    } {tag = "for_uni"}
    // A loop whose trip count differs between threads exits with different
    // values, but its induction variable is the same for the threads that
    // are still iterating.
    // expected-remark @below {{uniformity of "for_div": results = [divergent], execution = uniform}}
    %r3 = scf.for %j = %c0 to %tid step %c1 iter_args(%acc = %c0) -> index {
      // expected-remark @below {{uniformity of "iv_div_ub": results = [uniform], execution = divergent}}
      %next = arith.addi %acc, %j {tag = "iv_div_ub"} : index
      scf.yield %next : index
    } {tag = "for_div"}
    // expected-remark @below {{uniformity of "for_div_lb": results = [divergent], execution = uniform}}
    %r4 = scf.for %k = %tid to %n step %c1 iter_args(%acc = %c0) -> index {
      // expected-remark @below {{uniformity of "iv_div_lb": results = [divergent], execution = divergent}}
      %next = arith.addi %acc, %k {tag = "iv_div_lb"} : index
      scf.yield %next : index
    } {tag = "for_div_lb"}
    // The execution uniformity of an operation is the narrowest scope among
    // the control operands of its enclosing region branches.
    %sg = gpu.subgroup_id : index
    %sgcond = arith.cmpi ult, %sg, %n : index
    scf.if %sgcond {
      // expected-remark @below {{uniformity of "in_sg_if": results = [], execution = subgroup}}
      gpu.barrier {tag = "in_sg_if"}
      scf.if %cond {
        // expected-remark @below {{uniformity of "nested": results = [], execution = divergent}}
        gpu.barrier {tag = "nested"}
      }
    }
    gpu.return
  }
}

// -----

// scf.while: the condition steers the results.
gpu.module @while {
  gpu.func @kernel(%n: index) kernel {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tid = gpu.thread_id x
    // expected-remark @below {{uniformity of "while_div": results = [divergent], execution = uniform}}
    %r = scf.while (%i = %c0) : (index) -> index {
      %cond = arith.cmpi ult, %i, %tid : index
      scf.condition(%cond) %i : index
    } do {
    ^bb0(%i: index):
      // expected-remark @below {{uniformity of "in_while": results = [uniform], execution = divergent}}
      %next = arith.addi %i, %c1 {tag = "in_while"} : index
      scf.yield %next : index
    } attributes {tag = "while_div"}
    // expected-remark @below {{uniformity of "while_uni": results = [uniform], execution = uniform}}
    %s = scf.while (%i = %c0) : (index) -> index {
      %cond = arith.cmpi ult, %i, %n : index
      scf.condition(%cond) %i : index
    } do {
    ^bb0(%i: index):
      %next = arith.addi %i, %c1 : index
      scf.yield %next : index
    } attributes {tag = "while_uni"}
    gpu.return
  }
}

// -----

// Memory: a load from an address that is the same within a group observes the
// same value within that group, but no wider than the group that shares the
// memory: workgroup memory is a different memory in every workgroup and
// thread-private memory is a different memory in every thread. Any other
// operation that reads memory is divergent.
gpu.module @memory {
  memref.global "private" @shared : memref<8xf32, #gpu.address_space<workgroup>>
  gpu.func @kernel(%m: memref<16xf32>, %s: memref<16xf32, #gpu.address_space<workgroup>>) kernel {
    %c0 = arith.constant 0 : index
    %tid = gpu.thread_id x
    // expected-remark @below {{uniformity of "load_uni": results = [uniform], execution = uniform}}
    %v0 = memref.load %m[%c0] {tag = "load_uni"} : memref<16xf32>
    // expected-remark @below {{uniformity of "load_div": results = [divergent], execution = uniform}}
    %v1 = memref.load %m[%tid] {tag = "load_div"} : memref<16xf32>
    // A shared buffer is a different buffer in every workgroup, so reading it
    // at the same address says nothing across workgroups.
    // expected-remark @below {{uniformity of "load_shared": results = [workgroup], execution = uniform}}
    %v2 = memref.load %s[%c0] {tag = "load_shared"} : memref<16xf32, #gpu.address_space<workgroup>>
    // Reading a workgroup global has the same ceiling as reading an argument.
    %g = memref.get_global @shared : memref<8xf32, #gpu.address_space<workgroup>>
    // expected-remark @below {{uniformity of "load_global_shared": results = [workgroup], execution = uniform}}
    %v6 = memref.load %g[%c0] {tag = "load_global_shared"} : memref<8xf32, #gpu.address_space<workgroup>>
    // expected-remark @below {{uniformity of "alloca": results = [divergent], execution = uniform}}
    %p = memref.alloca() {tag = "alloca"} : memref<4xf32, #gpu.address_space<private>>
    // expected-remark @below {{uniformity of "load_private": results = [divergent], execution = uniform}}
    %v3 = memref.load %p[%c0] {tag = "load_private"} : memref<4xf32, #gpu.address_space<private>>
    // expected-remark @below {{uniformity of "vload": results = [uniform], execution = uniform}}
    %v4 = vector.load %m[%c0] {tag = "vload"} : memref<16xf32>, vector<4xf32>
    // expected-remark @below {{uniformity of "atomic": results = [divergent], execution = uniform}}
    %v5 = memref.atomic_rmw addf %v0, %m[%c0] {tag = "atomic"} : (f32, memref<16xf32>) -> f32
    // affine.load is modelled the same way as memref.load.
    // expected-remark @below {{uniformity of "affine_load_uni": results = [uniform], execution = uniform}}
    %v7 = affine.load %m[0] {tag = "affine_load_uni"} : memref<16xf32>
    // expected-remark @below {{uniformity of "affine_load_div": results = [divergent], execution = uniform}}
    %v8 = affine.load %m[symbol(%tid)] {tag = "affine_load_div"} : memref<16xf32>
    gpu.return
  }
}

// -----

// Kernel attributions: a workgroup buffer is shared by the workgroup, a private
// buffer is per thread, whatever is done with the value.
gpu.module @attributions {
  gpu.func @kernel(%n: index)
      workgroup(%ws: memref<4xf32, #gpu.address_space<workgroup>>)
      private(%ps: memref<4xf32, #gpu.address_space<private>>) kernel {
    // expected-remark @below {{uniformity of "ws_ptr": results = [workgroup], execution = uniform}}
    %a = memref.extract_aligned_pointer_as_index %ws : memref<4xf32, #gpu.address_space<workgroup>> -> index {tag = "ws_ptr"}
    // expected-remark @below {{uniformity of "ps_ptr": results = [divergent], execution = uniform}}
    %b = memref.extract_aligned_pointer_as_index %ps : memref<4xf32, #gpu.address_space<private>> -> index {tag = "ps_ptr"}
    gpu.return
  }
}

// -----

// An operation whose region captures a value from above without region
// control flow is not a function of its operands: it is divergent.
gpu.module @regions {
  gpu.func @kernel(%n: index) kernel {
    %tid = gpu.thread_id x
    // expected-remark @below {{uniformity of "generate": results = [divergent], execution = uniform}}
    %t = tensor.generate {
    ^bb0(%i: index):
      %v = arith.addi %i, %tid : index
      tensor.yield %v : index
    } {tag = "generate"} : tensor<4xindex>
    gpu.return
  }
}

// -----

// A func.func marked gpu.kernel receives the same arguments on every thread;
// the arguments of any other func.func are not known to be uniform.
gpu.module @func_kernels {
  func.func @kernel(%n: index) attributes {gpu.kernel} {
    // expected-remark @below {{uniformity of "kernel_arg": results = [uniform], execution = uniform}}
    %a = arith.addi %n, %n {tag = "kernel_arg"} : index
    return
  }
  func.func @device(%n: index) {
    // expected-remark @below {{uniformity of "device_arg": results = [divergent], execution = uniform}}
    %a = arith.addi %n, %n {tag = "device_arg"} : index
    return
  }
}

// -----

// Unstructured control flow: block arguments of non-entry blocks are tainted
// by every branch of the region, and so is their execution.
gpu.module @cfg {
  gpu.func @kernel(%n: index, %flag: i1) kernel {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tid = gpu.thread_id x
    %cond = arith.cmpi ult, %tid, %n : index
    cf.cond_br %cond, ^bb1(%c0 : index), ^bb1(%c1 : index)
  ^bb1(%x: index):
    // expected-remark @below {{uniformity of "phi": results = [divergent], execution = divergent}}
    %y = arith.addi %x, %c1 {tag = "phi"} : index
    gpu.return
  }
  gpu.func @uniform_branch(%n: index, %flag: i1) kernel {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    cf.cond_br %flag, ^bb1(%c0 : index), ^bb1(%c1 : index)
  ^bb1(%x: index):
    // expected-remark @below {{uniformity of "phi_uni": results = [uniform], execution = uniform}}
    %y = arith.addi %x, %c1 {tag = "phi_uni"} : index
    gpu.return
  }
}

// -----

// gpu.launch body arguments, and a warp region. The launch is under host
// control flow, which does not narrow the execution of its body.
func.func @launch(%n: index, %go: i1) {
  %c1 = arith.constant 1 : index
  scf.if %go {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %n, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %n, %sy = %c1, %sz = %c1) {
      // expected-remark @below {{uniformity of "in_launch": results = [], execution = uniform}}
      gpu.barrier {tag = "in_launch"}
      gpu.terminator
    }
  }
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %n, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %n, %sy = %c1, %sz = %c1) {
    // expected-remark @below {{uniformity of "launch_bx": results = [workgroup], execution = uniform}}
    %a = arith.addi %bx, %c1 {tag = "launch_bx"} : index
    // expected-remark @below {{uniformity of "launch_tx": results = [divergent], execution = uniform}}
    %b = arith.addi %tx, %c1 {tag = "launch_tx"} : index
    // expected-remark @below {{uniformity of "launch_gx": results = [uniform], execution = uniform}}
    %c = arith.addi %gx, %c1 {tag = "launch_gx"} : index
    gpu.terminator
  }
  return
}

func.func @warp(%laneid: index, %v: vector<32xf32>) {
  // expected-remark @below {{uniformity of "warp": results = [subgroup, divergent], execution = uniform}}
  %r:2 = gpu.warp_execute_on_lane_0(%laneid)[32] args(%v : vector<32xf32>) -> (f32, vector<1xf32>) {
  ^bb0(%arg: vector<32xf32>):
    // expected-remark @below {{uniformity of "in_warp": results = [uniform], execution = divergent}}
    %s = vector.reduction <add>, %arg {tag = "in_warp"} : vector<32xf32> into f32
    gpu.yield %s, %arg : f32, vector<32xf32>
  } {tag = "warp"}
  return
}

// -----

// affine.for behaves like scf.for: the induction variable is the join of the
// lower bound operands, and the upper bound only decides when a thread stops.
gpu.module @affine {
  gpu.func @kernel(%n: index) kernel {
    %tid = gpu.thread_id x
    affine.for %i = 0 to 8 {
      // expected-remark @below {{uniformity of "affine_iv": results = [uniform], execution = uniform}}
      %a = arith.addi %i, %i {tag = "affine_iv"} : index
    }
    affine.for %i = 0 to %tid {
      // The trip count varies across threads, but every thread still in the
      // loop is on the same iteration.
      // expected-remark @below {{uniformity of "affine_iv_div_ub": results = [uniform], execution = divergent}}
      %a = arith.addi %i, %i {tag = "affine_iv_div_ub"} : index
    }
    affine.for %i = %tid to 8 {
      // expected-remark @below {{uniformity of "affine_iv_div_lb": results = [divergent], execution = divergent}}
      %a = arith.addi %i, %i {tag = "affine_iv_div_lb"} : index
    }
    gpu.return
  }
}

// -----

// A divergent step taints the induction variable, like a divergent lower bound.
gpu.module @divergent_step {
  gpu.func @kernel(%n: index) kernel {
    %c0 = arith.constant 0 : index
    %tid = gpu.thread_id x
    %r = scf.for %i = %c0 to %n step %tid iter_args(%a = %c0) -> (index) {
      // expected-remark @below {{uniformity of "step_iv": results = [divergent], execution = divergent}}
      %b = arith.addi %i, %a {tag = "step_iv"} : index
      scf.yield %b : index
    }
    // expected-remark @below {{uniformity of "step_result": results = [divergent], execution = uniform}}
    %use = arith.addi %r, %r {tag = "step_result"} : index
    gpu.return
  }
}

// -----

// A value an implementation could not decide yet, because an operand had not
// been visited, must not be pinned to divergent: the loop result below is only
// known after the regions of the while have settled, and the shuffle that uses
// it is visited first.
gpu.module @late_operand {
  gpu.func @kernel(%n: index) kernel {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %r = scf.while (%i = %c0) : (index) -> index {
      %c = arith.cmpi ult, %i, %n : index
      scf.condition(%c) %i : index
    } do {
    ^bb0(%i: index):
      %nx = arith.addi %i, %c1 : index
      scf.yield %nx : index
    }
    %w = arith.index_cast %r : index to i32
    %off = arith.constant 1 : i32
    %width = arith.constant 32 : i32
    // The shuffled value is the loop result, which is only known once the
    // regions of the while have settled. Pinning it to divergent on the first
    // visit would be permanent, since the join is a meet.
    // expected-remark @below {{uniformity of "late_shuffle": results = [uniform, divergent], execution = uniform}}
    %s, %v = gpu.shuffle xor %w, %off, %width {tag = "late_shuffle"} : i32
    gpu.return
  }
}

// -----

// A callable the analysis is not told about receives divergent arguments: it
// joins every call site, and nothing says a function is only called from
// uniform control flow. Execution uniformity stops at the callable, so the
// body of a callee is reported as executed by the whole launch whatever the
// call site.
gpu.module @calls {
  func.func @callee(%x: index) -> index {
    // expected-remark @below {{uniformity of "callee_arg": results = [divergent], execution = uniform}}
    %a = arith.addi %x, %x {tag = "callee_arg"} : index
    return %a : index
  }
  gpu.func @kernel(%n: index) kernel {
    %tid = gpu.thread_id x
    // expected-remark @below {{uniformity of "call_uniform": results = [divergent], execution = uniform}}
    %r1 = func.call @callee(%n) {tag = "call_uniform"} : (index) -> index
    // expected-remark @below {{uniformity of "call_divergent": results = [divergent], execution = uniform}}
    %r2 = func.call @callee(%tid) {tag = "call_divergent"} : (index) -> index
    gpu.return
  }
}
