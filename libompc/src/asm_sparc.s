/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */

	.align  8
	.skip   16

	.type   LockWithLdstUB,#function
	.global LockWithLdstUB
LockWithLdstUB:
retry:
	ldstub  [%o0],%o1       ! atomic load store
	tst     %o1
	be      out
	nop
loop:
	ldub    [%o0],%o1       ! load and test
	tst     %o1
	bne     loop
	nop
	ba,a    retry
out:
	nop
	jmp     %o7+8   ! return
	nop

	.type   TestLockWithLdstUB,#function
	.global TestLockWithLdstUB
TestLockWithLdstUB:
	ldstub  [%o0],%o0       ! atomic load store
	jmp     %o7+8   ! return
	nop

	.type   UnlockWithLdstUB,#function
	.global UnlockWithLdstUB
UnlockWithLdstUB:
	stbar
	stb     %g0,[%o0]       ! clear lock
	jmp     %o7+8           ! return
	nop
