// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: llvm.func @fp_modifiers
llvm.func @fp_modifiers(%a: f32, %b: f32, %x: f16, %y: f16, %z: f16) {
  // CHECK: nvvm.addf {{.*}}, rnd = rp, sat = sat, ftz : f32
  %0 = nvvm.addf %a, %b, ftz, sat = sat, rnd = rp : f32

  // CHECK: nvvm.divf {{.*}}, ftz, approx : f32
  %1 = nvvm.divf %a, %b, approx, ftz : f32

  // CHECK: nvvm.fma {{.*}}, rnd = rn, relu, oob : f16
  %2 = nvvm.fma %x, %y, %z, rnd = rn, oob, relu : f16

  // CHECK: nvvm.sqrt {{.*}}, rnd = rz, ftz : f32
  %3 = nvvm.sqrt %a, rnd = rz, ftz : f32
  llvm.return
}
