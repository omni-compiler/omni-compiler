/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/*
 * regs:
 *	$0  ... return val
 *	$16 ... 1st arg
 *	$31 ... zero reg
 */

	.set noat
	.set noreorder
	
	.text

.align 3
.globl	__alpha_get_clock
.ent	__alpha_get_clock

__alpha_get_clock:
	rpcc	$0
	ret
.end	__alpha_get_clock


.align 3
.globl	__alpha_mbar
.ent	__alpha_mbar
__alpha_mbar:
	mb
	ret
.end	__alpha_mbar

.align 3
.globl	__alpha_spin_unlock
.ent	__alpha_spin_unlock
__alpha_spin_unlock:
	mb
	stl	$31, 0($16)		/* store zero to lock var */
	ret
.end	__alpha_spin_unlock

.align 3
.globl	__alpha_spin_test_lock
.globl	__alpha_spin_lock
.ent	__alpha_spin_test_lock
__alpha_spin_test_lock:
	ldl_l	$0, 0($16)		/* load lock var */
	blbs	$0, alreadyLocked	/* if the lock var is NOT zero, go to alreadyLocked */
	br	fromTest		/* otherwise go to fromTest */
__alpha_spin_lock:
retry:
	ldl_l	$0, 0($16)		/* load lock var */
	blbs	$0, loopBody		/* if the lock var is NOT zero, go to loopBody */
fromTest:
	mov	1, $0			/* otherwise set $0 to one */
	stl_c	$0, 0($16)		/* store one to the lock var */
	beq	$0, loopBody		/* if the store failed, go to loopBody */
	mb				/* othewise flush */
	ret				/* and return 1 */
loopBody:
	ldl	$0, 0($16)		/* load the lock var again */
	blbs	$0, loopBody		/* if the lock var is one, loop */
	br	retry			/* otherwise (means some one unlock the lock var) go to retry */

alreadyLocked:
	mov	0, $0			/* return 0 */
	ret
.end	__alpha_spin_test_lock
