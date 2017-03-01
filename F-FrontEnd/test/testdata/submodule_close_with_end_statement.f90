      MODULE m
        INTERFACE
          MODULE FUNCTION f(a)
            INTEGER :: f
            INTEGER :: a
          END
          MODULE FUNCTION g(a)
            INTEGER :: g
            INTEGER :: a
          END
        END INTERFACE
      END

      SUBMODULE(m) subm
      CONTAINS
        MODULE FUNCTION f(a)
          INTEGER :: f
          INTEGER :: a
          f = a + 1
        END
        MODULE PROCEDURE g
          g = a + 1
        END
      END
