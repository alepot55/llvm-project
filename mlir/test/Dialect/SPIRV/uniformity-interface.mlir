// RUN: mlir-opt -split-input-file -test-uniformity-analysis -verify-diagnostics %s

// The spirv dialect is transparent: an operation that computes a function of
// its operands joins them, a constant is uniform, and the arguments of a
// spirv.func are divergent, since nothing says they are the same on every
// invocation. Only spirv.* operations may appear in a spirv.module, so the
// sources of every scope are spirv operations themselves.
spirv.module Logical GLSL450 {
  spirv.func @transparent(%v: i32) "None" {
    // expected-remark @below {{uniformity of "arg": results = [divergent], execution = uniform}}
    %arg = spirv.IAdd %v, %v {tag = "arg"} : i32
    %c1 = spirv.Constant 1 : i32
    // expected-remark @below {{uniformity of "const": results = [uniform], execution = uniform}}
    %c = spirv.IAdd %c1, %c1 {tag = "const"} : i32
    %wg = spirv.GroupIAdd <Workgroup> <Reduce> %v : i32
    %sg = spirv.GroupNonUniformBroadcastFirst <Subgroup> %v : i32
    // Arithmetic joins its operands, which is the narrower scope.
    // expected-remark @below {{uniformity of "wg_wg": results = [workgroup], execution = uniform}}
    %a = spirv.IAdd %wg, %wg {tag = "wg_wg"} : i32
    // expected-remark @below {{uniformity of "wg_c": results = [workgroup], execution = uniform}}
    %b = spirv.IAdd %wg, %c1 {tag = "wg_c"} : i32
    // expected-remark @below {{uniformity of "wg_sg": results = [subgroup], execution = uniform}}
    %d = spirv.IAdd %wg, %sg {tag = "wg_sg"} : i32
    // expected-remark @below {{uniformity of "wg_div": results = [divergent], execution = uniform}}
    %e = spirv.IAdd %wg, %v {tag = "wg_div"} : i32
    // A load is not a function of its operands.
    %var = spirv.Variable : !spirv.ptr<i32, Function>
    // expected-remark @below {{uniformity of "load": results = [divergent], execution = uniform}}
    %l = spirv.Load "Function" %var {tag = "load"} : i32
    spirv.Return
  }
}

// -----

// The non-uniform group operations, whose scope is the subgroup.
spirv.module Logical GLSL450 {
  spirv.func @non_uniform(%v: i32, %p: i1) "None" {
    %c1 = spirv.Constant 1 : i32
    %c4 = spirv.Constant 4 : i32
    // Every invocation of the subgroup receives the same ballot mask.
    // expected-remark @below {{uniformity of "ballot": results = [subgroup], execution = uniform}}
    %ballot = spirv.GroupNonUniformBallot <Subgroup> %p {tag = "ballot"} : vector<4xi32>
    // expected-remark @below {{uniformity of "khr_ballot": results = [subgroup], execution = uniform}}
    %khr = spirv.KHR.SubgroupBallot %p {tag = "khr_ballot"} : vector<4xi32>
    // The position of a bit and the count of the bits of the mask are
    // functions of the mask; a scan of the bits counts up to the invocation's
    // own position.
    // expected-remark @below {{uniformity of "find_lsb": results = [subgroup], execution = uniform}}
    %lsb = spirv.GroupNonUniformBallotFindLSB <Subgroup> %ballot {tag = "find_lsb"} : vector<4xi32>, i32
    // expected-remark @below {{uniformity of "find_msb": results = [subgroup], execution = uniform}}
    %msb = spirv.GroupNonUniformBallotFindMSB <Subgroup> %khr {tag = "find_msb"} : vector<4xi32>, i32
    // expected-remark @below {{uniformity of "bit_count": results = [subgroup], execution = uniform}}
    %count = spirv.GroupNonUniformBallotBitCount <Subgroup> <Reduce> %ballot {tag = "bit_count"} : vector<4xi32> -> i32
    // expected-remark @below {{uniformity of "bit_count_scan": results = [divergent], execution = uniform}}
    %prefix = spirv.GroupNonUniformBallotBitCount <Subgroup> <ExclusiveScan> %ballot {tag = "bit_count_scan"} : vector<4xi32> -> i32
    // A broadcast makes one invocation's value the value of every invocation.
    // expected-remark @below {{uniformity of "broadcast": results = [subgroup], execution = uniform}}
    %bcast = spirv.GroupNonUniformBroadcast <Subgroup> %v, %c1 {tag = "broadcast"} : i32, i32
    // expected-remark @below {{uniformity of "broadcast_first": results = [subgroup], execution = uniform}}
    %first = spirv.GroupNonUniformBroadcastFirst <Subgroup> %v {tag = "broadcast_first"} : i32
    // Exactly one invocation is elected.
    // expected-remark @below {{uniformity of "elect": results = [divergent], execution = uniform}}
    %elect = spirv.GroupNonUniformElect <Subgroup> {tag = "elect"} : i1
    // A reduction is the same on every invocation; a scan is not, even of a
    // uniform value; a clustered reduction reduces within each cluster.
    // expected-remark @below {{uniformity of "reduce": results = [subgroup], execution = uniform}}
    %reduce = spirv.GroupNonUniformIAdd <Subgroup> <Reduce> %v {tag = "reduce"} : i32 -> i32
    // expected-remark @below {{uniformity of "scan": results = [divergent], execution = uniform}}
    %scan = spirv.GroupNonUniformIAdd <Subgroup> <InclusiveScan> %c1 {tag = "scan"} : i32 -> i32
    // expected-remark @below {{uniformity of "clustered": results = [divergent], execution = uniform}}
    %clustered = spirv.GroupNonUniformIAdd <Subgroup> <ClusteredReduce> %v cluster_size(%c4) {tag = "clustered"} : i32, i32 -> i32
    // expected-remark @below {{uniformity of "logical_and": results = [subgroup], execution = uniform}}
    %land = spirv.GroupNonUniformLogicalAnd <Subgroup> <Reduce> %p {tag = "logical_and"} : i1 -> i1
    // A property of the whole subgroup is delivered to every invocation.
    // expected-remark @below {{uniformity of "all": results = [subgroup], execution = uniform}}
    %all = spirv.GroupNonUniformAll <Subgroup> %p {tag = "all"} : i1
    // expected-remark @below {{uniformity of "any": results = [subgroup], execution = uniform}}
    %any = spirv.GroupNonUniformAny <Subgroup> %p {tag = "any"} : i1
    // expected-remark @below {{uniformity of "all_equal": results = [subgroup], execution = uniform}}
    %eq = spirv.GroupNonUniformAllEqual <Subgroup> %v {tag = "all_equal"} : i32, i1
    spirv.Return
  }
}

// -----

// A shuffle, a rotation or a quad swap permutes the copies of a value across
// the invocations: a divergent value stays divergent and a uniform value
// uniform, whatever the index, and a value that is the same within the
// workgroup is still the same after a permutation within the subgroup.
spirv.module Logical GLSL450 {
  spirv.func @permutations(%v: i32, %id: i32) "None" {
    %c1 = spirv.Constant 1 : i32
    %c4 = spirv.Constant 4 : i32
    %wg = spirv.GroupIAdd <Workgroup> <Reduce> %v : i32
    // expected-remark @below {{uniformity of "shuffle_div": results = [divergent], execution = uniform}}
    %s0 = spirv.GroupNonUniformShuffle <Subgroup> %v, %id {tag = "shuffle_div"} : i32, i32
    // expected-remark @below {{uniformity of "shuffle_uni": results = [uniform], execution = uniform}}
    %s1 = spirv.GroupNonUniformShuffle <Subgroup> %c1, %id {tag = "shuffle_uni"} : i32, i32
    // expected-remark @below {{uniformity of "shuffle_xor_wg": results = [workgroup], execution = uniform}}
    %s2 = spirv.GroupNonUniformShuffleXor <Subgroup> %wg, %id {tag = "shuffle_xor_wg"} : i32, i32
    // expected-remark @below {{uniformity of "shuffle_up_div": results = [divergent], execution = uniform}}
    %s3 = spirv.GroupNonUniformShuffleUp <Subgroup> %v, %c1 {tag = "shuffle_up_div"} : i32, i32
    // expected-remark @below {{uniformity of "shuffle_down_uni": results = [uniform], execution = uniform}}
    %s4 = spirv.GroupNonUniformShuffleDown <Subgroup> %c1, %c1 {tag = "shuffle_down_uni"} : i32, i32
    // expected-remark @below {{uniformity of "rotate_div": results = [divergent], execution = uniform}}
    %r0 = spirv.GroupNonUniformRotateKHR <Subgroup> %v, %c1 {tag = "rotate_div"} : i32, i32 -> i32
    // expected-remark @below {{uniformity of "rotate_uni": results = [uniform], execution = uniform}}
    %r1 = spirv.GroupNonUniformRotateKHR <Workgroup> %c1, %c1, cluster_size(%c4) {tag = "rotate_uni"} : i32, i32, i32 -> i32
    // expected-remark @below {{uniformity of "quad_swap_wg": results = [workgroup], execution = uniform}}
    %q = spirv.GroupNonUniformQuadSwap <Subgroup> <Horizontal> %wg {tag = "quad_swap_wg"} : i32
    spirv.Return
  }
}

// -----

// The uniform group operations, whose scope is the workgroup or the subgroup.
// These are pure, so without the interface the transparent rule would infer
// the scan of a uniform value to be uniform, while each invocation holds its
// own prefix.
spirv.module Logical GLSL450 {
  spirv.func @group(%v: i32, %f: f32, %id: i32) "None" {
    %c1 = spirv.Constant 1 : i32
    // expected-remark @below {{uniformity of "wg_reduce": results = [workgroup], execution = uniform}}
    %wg = spirv.GroupIAdd <Workgroup> <Reduce> %v {tag = "wg_reduce"} : i32
    // expected-remark @below {{uniformity of "sg_reduce": results = [subgroup], execution = uniform}}
    %sg = spirv.GroupFAdd <Subgroup> <Reduce> %f {tag = "sg_reduce"} : f32
    // expected-remark @below {{uniformity of "wg_scan": results = [divergent], execution = uniform}}
    %scan = spirv.GroupIAdd <Workgroup> <ExclusiveScan> %c1 {tag = "wg_scan"} : i32
    // expected-remark @below {{uniformity of "khr_reduce": results = [workgroup], execution = uniform}}
    %mul = spirv.KHR.GroupIMul <Workgroup> <Reduce> %v {tag = "khr_reduce"} : i32
    // expected-remark @below {{uniformity of "wg_broadcast": results = [workgroup], execution = uniform}}
    %b0 = spirv.GroupBroadcast <Workgroup> %v, %id {tag = "wg_broadcast"} : i32, i32
    // expected-remark @below {{uniformity of "sg_broadcast": results = [subgroup], execution = uniform}}
    %b1 = spirv.GroupBroadcast <Subgroup> %v, %id {tag = "sg_broadcast"} : i32, i32
    spirv.Return
  }
}
