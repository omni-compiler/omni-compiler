      MODULE m
        INTERFACE
           MODULE FUNCTION f(a)
             INTEGER(KIND=8) :: f
             INTEGER(KIND=8) :: a
           END FUNCTION f
        END INTERFACE
      END MODULE m

      SUBMODULE(m) subm
      CONTAINS
        MODULE PROCEDURE f
          f = a + 1
        END PROCEDURE g
      END SUBMODULE subm
