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
        MODULE FUNCTION f(a)
          INTEGER(KIND=8) :: f
          INTEGER(KIND=8) :: a

          f = a + 1
        END FUNCTION f
        MODULE SUBROUTINE g(a)
          REAL(KIND=8) :: a
          PRINT *, a
        END SUBROUTINE g
      END MODULE m

      PROGRAM main
        USE m
      END PROGRAM MAIN
