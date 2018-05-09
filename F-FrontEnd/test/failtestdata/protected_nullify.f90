      MODULE m
        INTEGER, POINTER, PROTECTED :: v(:)
      END MODULE m

      PROGRAM main
        USE m
        NULLIFY (v)
      END PROGRAM main
