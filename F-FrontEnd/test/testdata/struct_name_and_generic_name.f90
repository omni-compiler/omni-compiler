      MODULE m
        TYPE t
          REAL :: i
        END TYPE t
       CONTAINS
        FUNCTION f(a)
          COMPLEX :: a
          TYPE(t) :: f
          f%i = REAL(a) + AIMAG(a)
        END FUNCTION f
      END MODULE m

      PROGRAM main
        USE m
        INTERFACE t
          MODULE PROCEDURE :: f
        END INTERFACE t

        TYPE(t) :: v
        v = t(a=(1.0,2.0))
        PRINT *, v%i
      END PROGRAM main

 
