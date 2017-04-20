      PROGRAM main
        IMPLICIT NONE
        INTEGER, DIMENSION(3,3) :: A
        INTEGER, DIMENSION(3,3) :: B = RESHAPE((/1,2,3,4,5,6,7,8,9/), (/3,3/))
        INTEGER :: i, j
        FORALL (i = 1:3)
          FORALL (j = 1:3)
            A(i, j) = B(i, j)
          END FORALL
        END FORALL
        PRINT *, A(:,1)
        PRINT *, A(:,2)
        PRINT *, A(:,3)
      END PROGRAM main
