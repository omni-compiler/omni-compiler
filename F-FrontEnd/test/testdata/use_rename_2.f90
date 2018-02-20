      MODULE m
        INTEGER :: v
        INTEGER :: w
      END MODULE

      PROGRAM main
        USE m, u => v
        IMPLICIT NONE
        PRINT *, w
      END
