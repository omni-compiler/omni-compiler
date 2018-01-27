      MODULE m
        INTEGER, POINTER, PROTECTED :: n
      END MODULE m

      PROGRAM main
        USE m
        n = 1
      END PROGRAM main
