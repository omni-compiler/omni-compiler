      MODULE m
        INTERFACE
           MODULE FUNCTION f(a)
             INTEGER(KIND=8) :: f
             INTEGER(KIND=8) :: a
           END FUNCTION f
           MODULE SUBROUTINE g(a)
             REAL(KIND=8) :: a
           END SUBROUTINE g
        END INTERFACE
      CONTAINS
        MODULE PROCEDURE f
          f = a + 1
        END PROCEDURE f
        MODULE PROCEDURE g
          PRINT *, a
        END PROCEDURE g
      END MODULE m

      PROGRAM main
        USE m
      END PROGRAM MAIN
