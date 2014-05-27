C ********************************************************************
C                OpenMP Fortran API Test Suite
C                -----------------------------
C
C Test Name    : gtnthd001
C
C Summary      : Simple example of determining number of threads
C
C Description  : F77 version of example A.14 from specification.
C
C Verification : Execution self-checks verify results but not work
C                sharing.
C
C Origin       : Example A.14 from OpenMP Fortran API specification
C
C Keywords     : F77, parallel, parallel do, schedule
C Keywords     : omp_get_num_threads, omp_get_thread_num
C
C Source Form  : Fixed
C
C Last Changed : $Date: 2004/02/06 18:15:44 $
C
C ********************************************************************

      SUBROUTINE GNT001S()
      INTEGER OMP_GET_NUM_THREADS, OMP_GET_THREAD_NUM
      EXTERNAL OMP_GET_NUM_THREADS, OMP_GET_THREAD_NUM
C$OMP PARALLEL
      NP = OMP_GET_NUM_THREADS()
C$OMP DO SCHEDULE(STATIC)
      DO 100 I=0,NP-1
        CALL WORK(I)
  100 CONTINUE
C$OMP END DO
C$OMP END PARALLEL
C$OMP PARALLEL PRIVATE(I)
      I = OMP_GET_THREAD_NUM()
      CALL WORK2(I)
C$OMP END PARALLEL
      END

      SUBROUTINE WORK(I)
      COMMON /A/IA(1024)
      COMMON /IDX/MAXI
      IF ( I .LT. MAXI ) THEN
        IA(I+1) = I+1
      ENDIF
      END

      SUBROUTINE WORK2(I)
      COMMON /B/IB(1024)
      COMMON /IDX/MAXI
      IF ( I .LT. MAXI ) THEN
        IB(I+1) = I+1
      ENDIF
      END

      PROGRAM GNT001
      INTEGER I, ERRORS
      PARAMETER(N = 1024)
      COMMON /A/IA(N)
      COMMON /B/IB(N)
      COMMON /IDX/MAXI
      MAXI = N
      DO 200 I=1,N
        IA(I) = -1
        IB(I) = -1
  200 CONTINUE
c      CALL GNT001S(N)
      CALL GNT001S
C       Determine last elements modified
      LASTA = 0
      LASTB = 0
      DO 300 I=1,N
        IF (IA(I) .GE. 0) LASTA = I
        IF (IB(I) .GE. 0) LASTB = I
  300 CONTINUE
C       Infer number of threads
      ERRORS = 0
C       Should be same number modified in A and B
      IF ( LASTA .NE. LASTB ) THEN
        ERRORS = ERRORS + 1
        PRINT *,'gtnthd001 - Expected LASTA = LASTB, observed: ',
     1	        LASTA, LASTB
      ELSE
        PRINT *,'gtnthd001 - Apparent number threads = ', LASTA
      ENDIF
C       Check values in A
      DO 400 I=1,LASTA
        IF (IA(I) .NE. I) THEN
	  PRINT *,'gtnthd001 - Expected IA(', I, ') = ', I,
     1            ', OBSERVED ', IA(I)	
	ENDIF
  400 CONTINUE
C       Check values in B
      DO 500 I=1,LASTB
        IF (IB(I) .NE. I) THEN
	  PRINT *,'gtnthd001 - Expected IB(', I, ') = ', I,
     1            ', OBSERVED ', IB(I)	
	ENDIF
  500 CONTINUE
      IF (ERRORS .EQ. 0) THEN
        WRITE (*,'(A)') 'gtnthd001 PASSED'
      ELSE
        WRITE (*,'(A)') 'gtnthd001 FAILED'
      ENDIF
      END
