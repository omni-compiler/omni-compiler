C ********************************************************************
C                OpenMP Fortran API Test Suite
C                -----------------------------
C
C Test Name    : flsh001
C
C Summary      : Simple example of flush directive
C
C Description  : F77 version of example A.13 from specification.
C
C Verification : Execution self-checks verify results but not work
C                sharing.
C
C Origin       : Example A.13 from OpenMP Fortran API specification
C
C Keywords     : F77, parallel, flush, barrier, shared, default
C Keywords     : omp_get_thread_num
C
C Source Form  : Fixed
C
C Last Changed : $Date: 2004/02/06 18:15:44 $
C
C ********************************************************************

      SUBROUTINE FLSH001S(N, ICOUNT)
      DIMENSION ICOUNT(N), ISYNC(N)
      INTEGER   OMP_GET_THREAD_NUM
      EXTERNAL  OMP_GET_THREAD_NUM
C$OMP PARALLEL DEFAULT(PRIVATE) SHARED(ISYNC, ICOUNT)
      IAM = 1 + OMP_GET_THREAD_NUM()
      ISYNC(IAM) = 0
C$OMP BARRIER
      CALL WORK(ICOUNT, IAM)
      IF (IAM .GT. 1 ) THEN
C       WAIT TILL NEIGHBOR IS DONE
	NEIGH = IAM - 1
  100   CONTINUE
C$OMP   FLUSH(ISYNC)
        IF (ISYNC(NEIGH) .EQ. 0) GO TO 100
        ICOUNT(IAM) = ICOUNT(IAM) + ICOUNT(NEIGH)
      ENDIF
C     I AM DONE WITH MY WORK, SYNCHRONIZE WITH OTHER NEIGHBOR
      ISYNC(IAM) = 1
C$OMP FLUSH(ISYNC)
C$OMP END PARALLEL
      END

      SUBROUTINE WORK(IA, I)
      DIMENSION IA(*)
      IA(I) = I
      END

      PROGRAM FLSH001
      PARAMETER(N=1024)
      INTEGER I, ICOUNT(N), ERRORS
      DO 200 I=1,N
         ICOUNT(I) = -1
  200 CONTINUE
      CALL FLSH001S(N, ICOUNT)
C       Determine last element modified
      NT = 0
      DO 300 I=1,N
        IF (ICOUNT(I) .LT. 0) GOTO 400
        NT = I
  300 CONTINUE
  400 CONTINUE
      ERRORS = 0
      IEXPECT = 0
      DO 500 I=1,NT
	IEXPECT = IEXPECT + I
	IF (ICOUNT(I) .NE. IEXPECT) THEN
	  PRINT *, 'flsh001 - EXPECTED ICOUNT(', I, ') = ', IEXPECT,
     1                    ', OBSERVED ', ICOUNT(I)
	ENDIF
  500 CONTINUE
      PRINT *, 'flsh001 - Apparent number of threads is ', NT
      IF (ERRORS .EQ. 0) THEN
        WRITE (*,'(A)') 'flsh001 PASSED'
      ELSE
        WRITE (*,'(A)') 'flsh001 FAILED'
      ENDIF
      END
