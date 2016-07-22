      PROGRAM main
        USE ISO_FORTRAN_ENV

        INTEGER :: v = 1
        CHARACTER :: c = 'a'
        TYPE(LOCK_TYPE) :: a[*]
        LOGICAL :: ACQUIRED_LOCK = .TRUE.

        LOCK (a, ACQUIRED_LOCK=ACQUIRED_LOCK, STAT=v, ERRMSG=c)
        UNLOCK (a, STAT=v, ERRMSG=c)
      END PROGRAM main
