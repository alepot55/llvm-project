// RUN: mlir-opt -split-input-file -test-uniformity-analysis -verify-diagnostics %s

// affine.for describes its induction variable like scf.for: the join of the
// lower bound operands, since the step is a constant. The upper bound only
// decides when a thread stops iterating.
func.func @induction_variable() {
  %n = test.with_uniformity {scope = "uniform"} : index
  %tid = test.with_uniformity {scope = "divergent"} : index
  affine.for %i = 0 to 8 {
    // expected-remark @below {{uniformity of "iv": results = [uniform], execution = uniform}}
    %a = arith.addi %i, %i {tag = "iv"} : index
  }
  affine.for %i = 0 to %n {
    // expected-remark @below {{uniformity of "iv_uni_ub": results = [uniform], execution = uniform}}
    %a = arith.addi %i, %i {tag = "iv_uni_ub"} : index
  }
  affine.for %i = 0 to %tid {
    // expected-remark @below {{uniformity of "iv_div_ub": results = [uniform], execution = divergent}}
    %a = arith.addi %i, %i {tag = "iv_div_ub"} : index
  }
  affine.for %i = %tid to 8 {
    // expected-remark @below {{uniformity of "iv_div_lb": results = [divergent], execution = divergent}}
    %a = arith.addi %i, %i {tag = "iv_div_lb"} : index
  }
  return
}
