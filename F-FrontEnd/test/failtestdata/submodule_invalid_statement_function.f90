      MODULE m
        INTERFACE
          MODULE SUBROUTINE sub()
          END SUBROUTINE
        END INTERFACE
      END MODULE m

      SUBMODULE(m) subm
        INTEGER x, y
        INTEGER g
        g(x, y) = x ** y + 1
      END SUBMODULE subm

      PROGRAM main
        USE m
      END PROGRAM main
