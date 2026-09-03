// RUN: mlir-opt -split-input-file -test-uniformity-analysis -verify-diagnostics %s

// The induction variable of an scf.for is the join of the lower bound and the
// step. The upper bound only decides when a thread stops iterating: it taints
// the results and the execution of the body, not the induction variable the
// threads still iterating observe.
func.func @induction_variable() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = test.with_uniformity {scope = "uniform"} : index
  %tid = test.with_uniformity {scope = "divergent"} : index
  // expected-remark @below {{uniformity of "for_uni": results = [uniform], execution = uniform}}
  %r0 = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0) -> index {
    // expected-remark @below {{uniformity of "iv_uni": results = [uniform], execution = uniform}}
    %next = arith.addi %acc, %i {tag = "iv_uni"} : index
    scf.yield %next : index
  } {tag = "for_uni"}
  // expected-remark @below {{uniformity of "for_div_ub": results = [divergent], execution = uniform}}
  %r1 = scf.for %i = %c0 to %tid step %c1 iter_args(%acc = %c0) -> index {
    // expected-remark @below {{uniformity of "iv_div_ub": results = [uniform], execution = divergent}}
    %next = arith.addi %acc, %i {tag = "iv_div_ub"} : index
    scf.yield %next : index
  } {tag = "for_div_ub"}
  // expected-remark @below {{uniformity of "for_div_lb": results = [divergent], execution = uniform}}
  %r2 = scf.for %i = %tid to %n step %c1 iter_args(%acc = %c0) -> index {
    // expected-remark @below {{uniformity of "iv_div_lb": results = [divergent], execution = divergent}}
    %next = arith.addi %acc, %i {tag = "iv_div_lb"} : index
    scf.yield %next : index
  } {tag = "for_div_lb"}
  // expected-remark @below {{uniformity of "for_div_step": results = [divergent], execution = uniform}}
  %r3 = scf.for %i = %c0 to %n step %tid iter_args(%acc = %c0) -> index {
    // expected-remark @below {{uniformity of "iv_div_step": results = [divergent], execution = divergent}}
    %next = arith.addi %acc, %i {tag = "iv_div_step"} : index
    scf.yield %next : index
  } {tag = "for_div_step"}
  return
}
