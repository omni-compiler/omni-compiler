      PROGRAM main
        USE ISO_FORTRAN_ENV
        TYPE s
          TYPE(LOCK_TYPE) :: a
        END TYPE s
        TYPE(s) :: v[*]
        LOCK ( v%a )
        UNLOCK ( v%a )
      END PROGRAM main
