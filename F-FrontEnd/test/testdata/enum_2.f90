      MODULE m
        PUBLIC :: a, b, i
        PRIVATE :: c
        ENUM, BIND(C)
          ENUMERATOR a, b, c
        END ENUM
        INTEGER, PARAMETER :: i = 10
      END MODULE m

      MODULE m2
        USE m
        PRIVATE :: i
      END MODULE m2

      PROGRAM main
        USE m2
        IMPLICIT NONE
        INTEGER :: v = b
      END PROGRAM main
