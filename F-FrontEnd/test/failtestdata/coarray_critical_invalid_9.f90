      PROGRAM main
        USE ISO_FORTRAN_ENV
        TYPE(LOCK_TYPE) :: a[*]
        CRITICAL
          UNLOCK ( a )
        END CRITICAL
      END PROGRAM main
