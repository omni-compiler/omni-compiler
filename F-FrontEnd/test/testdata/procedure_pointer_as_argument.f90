      MODULE m
       CONTAINS
        SUBROUTINE sub(g)
          PROCEDURE(f) :: g ! If this line does not exsit, F_Front works
          PRINT*, f(2) /= 'aa'
        END SUBROUTINE
        FUNCTION f(i)
          CHARACTER(i)::f
        END FUNCTION
      END MODULE m
