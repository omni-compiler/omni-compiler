      PROGRAM main
        USE ISO_FORTRAN_ENV
        TYPE(LOCK_TYPE) :: a[*]
        LOCK ( a )
        UNLOCK ( a )
      END PROGRAM main
