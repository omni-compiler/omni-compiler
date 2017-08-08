      MODULE m
        INTEGER, POINTER, PROTECTED :: v
      END MODULE m

      PROGRAM main
        USE m
        INTEGER, TARGET :: w
        v => w
      END PROGRAM main
