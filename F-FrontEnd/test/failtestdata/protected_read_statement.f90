      MODULE m
        INTEGER, PROTECTED :: v
      END MODULE m

      PROGRAM main
        USE m
        READ (6, *) v
      END PROGRAM main
