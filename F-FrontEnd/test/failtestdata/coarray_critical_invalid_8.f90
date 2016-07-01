      PROGRAM main
        USE ISO_FORTRAN_ENV
        TYPE(LOCK_TYPE) :: a[*]
        CRITICAL
          LOCK ( a )
        END CRITICAL
      END PROGRAM main
