      MODULE m
        CHARACTER, PROTECTED :: v
      END MODULE m

      PROGRAM main
        USE m
        WRITE (UNIT=v, FMT=*) "hello"
      END PROGRAM main
