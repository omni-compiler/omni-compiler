      MODULE generic_procedure
        INTERFACE f
           MODULE PROCEDURE f1
           MODULE PROCEDURE f2
        END INTERFACE f
      CONTAINS
        INTEGER FUNCTION f1(a)
          INTEGER :: a
          f1 = 1
        END FUNCTION f1
        INTEGER FUNCTION f2(a)
          LOGICAL :: a
          f2 = 1
        END FUNCTION f2
      END MODULE generic_procedure
