      MODULE m
        INTERFACE
          MODULE SUBROUTINE sub()
          END SUBROUTINE
        END INTERFACE
      END MODULE m

      SUBMODULE(m) subm
        PRIVATE :: i
      END SUBMODULE subm

      PROGRAM main
        USE m
      END PROGRAM main
