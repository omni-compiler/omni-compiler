      FUNCTION f(a)
        INTEGER :: f
        INTEGER :: a
      END

      FUNCTION g(a)
        INTEGER :: f
        INTEGER :: a
      CONTAINS
        SUBROUTINE sub()
        END
      END

      SUBROUTINE sub1()
      END

      SUBROUTINE sub2()
      CONTAINS
        SUBROUTINE sub()
        END
      END

      MODULE m1
        INTERFACE
          MODULE SUBROUTINE sub()
          END
        END INTERFACE
      END

      MODULE m2
      CONTAINS
        MODULE SUBROUTINE sub()
        END
      END

      BLOCK DATA
      END

      SUBMODULE(m1) subm1
      CONTAINS
        MODULE PROCEDURE sub
        END
      END

      SUBMODULE(m2) subm2
      END

      PROGRAM main
      CONTAINS
        SUBROUTINE sub()
        END
      END
