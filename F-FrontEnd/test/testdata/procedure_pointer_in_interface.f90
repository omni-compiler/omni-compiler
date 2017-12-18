      MODULE m
        IMPLICIT NONE

        INTERFACE
          INTEGER FUNCTION f()
          END FUNCTION f
        END INTERFACE

        PROCEDURE(f) :: x

        INTERFACE g
          PROCEDURE x
        END INTERFACE g

      END MODULE m
