      MODULE m
       CONTAINS

        IMPURE ELEMENTAL FUNCTION f()
          INTEGER :: f
        END FUNCTION f

      END MODULE m

      PROGRAM main
        USE m
      END PROGRAM main
