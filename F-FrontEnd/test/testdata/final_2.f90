  MODULE m
   IMPLICIT NONE

   TYPE :: t
     INTEGER :: v
   CONTAINS
     FINAL :: fin1, fin2
   END TYPE t

 CONTAINS

   SUBROUTINE fin1(this)
     TYPE (t),INTENT(INOUT) :: this
     PRINT *, 'call fin1'
     RETURN
   END SUBROUTINE fin1
   SUBROUTINE fin2(this)
     TYPE (t),INTENT(INOUT), DIMENSION(:) :: this
     PRINT *, 'call fin2'
     RETURN
   END SUBROUTINE fin2

END MODULE m
