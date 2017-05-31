  MODULE m_test
   IMPLICIT NONE

   TYPE :: t_test
     INTEGER, POINTER :: i(:)
   CONTAINS
     PROCEDURE,pass(bb)   :: calc_test
     PROCEDURE,pass       :: alloc_test
     PROCEDURE,pass(this) :: init_test
     FINAL                :: dealloc_test
   END TYPE t_test

 CONTAINS

   INTEGER FUNCTION calc_test(bb)
     CLASS (t_test), INTENT(IN) :: bb
     calc_test = SUM(bb%i)
     RETURN
   END FUNCTION calc_test

   SUBROUTINE alloc_test(this, n)
     CLASS (t_test), INTENT(INOUT) :: this
     INTEGER, INTENT(IN) :: n
     ALLOCATE(this%i(n) )
     RETURN
   END SUBROUTINE alloc_test

   SUBROUTINE init_test(this)
    CLASS (t_test), INTENT(IN) :: this
    INTEGER :: k
    this%i = [(k,  k = 1, SIZE(this%i) )]
    RETURN
   END SUBROUTINE init_test

   SUBROUTINE dealloc_test(this)
     TYPE (t_test),INTENT(INOUT) :: this
     PRINT *, '=========== deallocated =============='
     DEALLOCATE(this%i)
     RETURN
   END SUBROUTINE dealloc_test

END MODULE m_test

PROGRAM test
  USE m_test
  TYPE (t_test), ALLOCATABLE :: tt
  ALLOCATE(tt)
  CALL tt%alloc_test(10**6)
  CALL tt%init_test()
  PRINT *, tt%calc_test()
  DEALLOCATE(tt)
  STOP
END PROGRAM test
