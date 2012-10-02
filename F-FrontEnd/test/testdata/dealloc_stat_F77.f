      subroutine dealloc_stat

      INTEGER :: ISTAT=0, N=10
      INTEGER, TARGET :: ISTAT1(1:10,1:20)
      INTEGER, POINTER :: IWORK(:)
      COMPLEX, POINTER :: WORK(:)
      REAL, POINTER :: RWORK(:)

      ALLOCATE(IWORK(MAX(1,5*N)), RWORK(MAX(1,7*N)), 
     + WORK(MAX(1,N)), STAT=ISTAT)

      DEALLOCATE(IWORK, RWORK, WORK, STAT=ISTAT1(1,1))

      end subroutine dealloc_stat

