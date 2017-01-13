      MODULE m
        INTERFACE
          MODULE SUBROUTINE sub()
          END SUBROUTINE
        END INTERFACE
      END MODULE m

      SUBMODULE(m) subm
 9      format (F10.3)
      END SUBMODULE subm

      PROGRAM main
        USE m
      END PROGRAM main
