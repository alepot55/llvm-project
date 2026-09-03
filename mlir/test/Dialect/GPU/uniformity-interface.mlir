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
    // In idx mode the second operand is the lane every lane reads from, so a
    // lane index that is the same across the subgroup makes even a divergent
    // value the same across the subgroup.
    // expected-remark @below {{uniformity of "shfl_idx": results = [subgroup, divergent], execution = uniform}}
    %s2, %valid2 = gpu.shuffle idx %v, %o, %w {tag = "shfl_idx"} : i32
    // expected-remark @below {{uniformity of "shfl_idx_div": results = [divergent, divergent], execution = uniform}}
    %s3, %valid3 = gpu.shuffle idx %v, %v, %w {tag = "shfl_idx_div"} : i32
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

// The execution uniformity of an operation is the narrowest scope among the
// control operands of its enclosing region branches: a barrier under
// subgroup-uniform control flow is reached by whole subgroups.
gpu.module @execution {
  gpu.func @kernel(%n: index) kernel {
    %tid = gpu.thread_id x
    %sg = gpu.subgroup_id : index
    %cond = arith.cmpi ult, %tid, %n : index
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
             threads(%tx, %ty, %tz) in (%sx = %n, %sy = %c1, %sz = %c1)
             workgroup(%ws: memref<4xf32, #gpu.address_space<workgroup>>)
             private(%ps: memref<4xf32, #gpu.address_space<private>>) {
    // expected-remark @below {{uniformity of "launch_bx": results = [workgroup], execution = uniform}}
    %a = arith.addi %bx, %c1 {tag = "launch_bx"} : index
    // expected-remark @below {{uniformity of "launch_tx": results = [divergent], execution = uniform}}
    %b = arith.addi %tx, %c1 {tag = "launch_tx"} : index
    // expected-remark @below {{uniformity of "launch_gx": results = [uniform], execution = uniform}}
    %c = arith.addi %gx, %c1 {tag = "launch_gx"} : index
    // The attributions of a launch are modelled like those of a kernel.
    // expected-remark @below {{uniformity of "launch_ws": results = [workgroup], execution = uniform}}
    %d = memref.extract_aligned_pointer_as_index %ws : memref<4xf32, #gpu.address_space<workgroup>> -> index {tag = "launch_ws"}
    // expected-remark @below {{uniformity of "launch_ps": results = [divergent], execution = uniform}}
    %e = memref.extract_aligned_pointer_as_index %ps : memref<4xf32, #gpu.address_space<private>> -> index {tag = "launch_ps"}
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

// The rest of the GPU operations that describe themselves through the
// interface, and a gpu.func that is not a kernel, whose arguments nothing says
// are the same on every thread.
gpu.module @more_sources {
  gpu.func @kernel(%p: i1) kernel {
    %tid = gpu.thread_id x
    %v = arith.index_cast %tid : index to i32
    // The ballot mask is the same for every lane of the subgroup.
    // expected-remark @below {{uniformity of "ballot": results = [subgroup], execution = uniform}}
    %b = gpu.ballot %p {tag = "ballot"} : i32
    // expected-remark @below {{uniformity of "cluster_block_id": results = [workgroup], execution = uniform}}
    %cb = gpu.cluster_block_id x {tag = "cluster_block_id"}
    // The global index contains the thread index.
    // expected-remark @below {{uniformity of "global_id": results = [divergent], execution = uniform}}
    %g = gpu.global_id x {tag = "global_id"}
    // A rotate reads another lane, so neither its value nor its validity is
    // the same across the subgroup.
    // expected-remark @below {{uniformity of "rotate": results = [divergent, divergent], execution = uniform}}
    %r, %rv = gpu.rotate %v, 1, 32 {tag = "rotate"} : i32
    // A broadcast makes one lane's value the value of every lane.
    // expected-remark @below {{uniformity of "broadcast": results = [subgroup], execution = uniform}}
    %bc = gpu.subgroup_broadcast %v, first_active_lane {tag = "broadcast"} : i32
    gpu.return
  }
  gpu.func @not_a_kernel(%n: index) {
    // expected-remark @below {{uniformity of "non_kernel_arg": results = [divergent], execution = uniform}}
    %a = arith.addi %n, %n {tag = "non_kernel_arg"} : index
    gpu.return
  }
}

// -----

// What the transparent dialect list costs. An operation of a dialect outside
// it defines divergent values even when it is pure and even when it is in
// truth uniform: the analysis has no way to tell the block size from the
// thread index, since either could read thread identity.
gpu.module @not_transparent {
  gpu.func @kernel() kernel {
    // expected-remark @below {{uniformity of "nvvm_ntid": results = [divergent], execution = uniform}}
    %ntid = nvvm.read.ptx.sreg.ntid.x {tag = "nvvm_ntid"} : i32
    // expected-remark @below {{uniformity of "nvvm_tid": results = [divergent], execution = uniform}}
    %tid = nvvm.read.ptx.sreg.tid.x {tag = "nvvm_tid"} : i32
    gpu.return
  }
}
