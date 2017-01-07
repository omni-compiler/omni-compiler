      MODULE m1
        INTEGER :: i
        INTERFACE
          MODULE FUNCTION f(a)
            INTEGER :: f
            INTEGER :: a
          END FUNCTION f
        END INTERFACE
      END MODULE m1

      SUBMODULE(m1) subm
        INTEGER :: g
      END SUBMODULE subm

      SUBMODULE(m1:subm) subsubm
        INTEGER :: h
      CONTAINS
        MODULE FUNCTION f(a)
          INTEGER :: f
          INTEGER :: a
        END FUNCTION
      END SUBMODULE subsubm

      PROGRAM main
        USE m
      END PROGRAM main
