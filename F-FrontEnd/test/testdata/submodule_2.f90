      MODULE m2
        COMPLEX :: i
        INTERFACE
          MODULE SUBROUTINE sub()
          END SUBROUTINE sub
       END INTERFACE
      END MODULE m2

      SUBMODULE(m2) subm
      CONTAINS
        MODULE SUBROUTINE sub()
          REAL :: r
          r = REAL(i)
        END SUBROUTINE SUB
      END SUBMODULE subm

      PROGRAM main
        USE m2
      END PROGRAM MAIN
