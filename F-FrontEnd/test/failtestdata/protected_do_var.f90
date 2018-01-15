      MODULE m
        INTEGER, PROTECTED :: v
      END MODULE m

      PROGRAM main
        USE m
        DO v = 1,10
          WRITE (*,*) v
        END DO
      END PROGRAM main
