      PROGRAM main
        IMPLICIT NONE
        INTEGER, DIMENSION(3,3) :: A
        INTEGER, DIMENSION(3,3) :: B = RESHAPE((/1,2,3,4,5,6,7,8,9/), (/3,3/))
        INTEGER :: i, j
        FORALL (i = 1:3, j = 1:3, .TRUE.)
          A(i, j) = B(i, j)
          CALL sub()
        END FORALL
       CONTAINS
        SUBROUTINE sub()
        END SUBROUTINE
      END PROGRAM main
