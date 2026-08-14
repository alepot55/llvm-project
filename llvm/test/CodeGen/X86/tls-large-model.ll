; RUN: llc < %s -mtriple=x86_64-linux-gnu -relocation-model=pic \
; RUN:   -code-model=large | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-linux-gnu -relocation-model=pic \
; RUN:   -code-model=large -filetype=obj -o - | llvm-readobj -r - | \
; RUN:   FileCheck %s --check-prefix=RELOC

; RELOC-DAG: R_X86_64_GOTPC64 _GLOBAL_OFFSET_TABLE_
; RELOC-DAG: R_X86_64_GOTPC64 _GLOBAL_OFFSET_TABLE_
; RELOC-DAG: R_X86_64_TLSGD gd
; RELOC-DAG: R_X86_64_PLTOFF64 __tls_get_addr
; RELOC-DAG: R_X86_64_PLTOFF64 __tls_get_addr
; RELOC-DAG: R_X86_64_TLSLD ld

@gd = external thread_local global i32
@ld = external thread_local(localdynamic) global i32

define i32 @load_gd() {
; CHECK-LABEL: load_gd:
; CHECK:       [[PB:\.L[^:]+]]:
; CHECK-NEXT:    leaq [[PB]](%rip), %r11
; CHECK-NEXT:    movabsq $_GLOBAL_OFFSET_TABLE_-[[PB]], %rax
; CHECK-NEXT:    addq %rax, %r11
; CHECK-NEXT:    leaq gd@TLSGD(%rip), %rdi
; CHECK-NEXT:    movabsq $__tls_get_addr@PLTOFF, %rax
; CHECK-NEXT:    addq %r11, %rax
; CHECK-NEXT:    callq *%rax
; CHECK-NEXT:    movl (%rax), %eax
; CHECK:         retq
  %v = load i32, ptr @gd
  ret i32 %v
}

define i32 @load_ld() {
; CHECK-LABEL: load_ld:
; CHECK:       [[PB:\.L[^:]+]]:
; CHECK-NEXT:    leaq [[PB]](%rip), %r11
; CHECK-NEXT:    movabsq $_GLOBAL_OFFSET_TABLE_-[[PB]], %rax
; CHECK-NEXT:    addq %rax, %r11
; CHECK-NEXT:    leaq ld@TLSLD(%rip), %rdi
; CHECK-NEXT:    movabsq $__tls_get_addr@PLTOFF, %rax
; CHECK-NEXT:    addq %r11, %rax
; CHECK-NEXT:    callq *%rax
; CHECK-NEXT:    movl (%rbx,%rax), %eax
; CHECK:         retq
  %v = load i32, ptr @ld
  ret i32 %v
}
