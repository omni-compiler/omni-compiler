      MODULE m
        CHARACTER, PROTECTED :: v
      END MODULE m

      PROGRAM main
        USE m
        WRITE (v, *) "hello"
      END PROGRAM main
