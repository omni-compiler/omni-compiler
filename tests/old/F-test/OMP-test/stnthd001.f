C ********************************************************************
C                OpenMP Fortran API Test Suite
C                -----------------------------
C
C Test Name    : stnthd001
C
C Summary      : Simple example of specifying number of threads
C
C Description  : F77 version of example A.11 from specification.
C
C Verification : Execution self-checks verify results but not work
C                sharing.
C
C Origin       : Example A.11 from OpenMP Fortran API specification
C
C Keywords     : F77, parallel, default, shared, omp_set_dynamic
C Keywords     : omp_set_num_threads, omp_get_thread_num
C
C Source Form  : Fixed
C
C Last Changed : $Date: 2004/02/06 18:15:44 $
C
C ********************************************************************

      SUBROUTINE SNT001S(X,NPOINTS)
      INTEGER OMP_GET_THREAD_NUM
      EXTERNAL OMP_GET_THREAD_NUM
      INTEGER X(NPOINTS)
      CALL OMP_SET_DYNAMIC(.FALSE.)
      CALL OMP_SET_NUM_THREADS(16)
C$OMP PARALLEL DEFAULT(PRIVATE) SHARED(X,NPOINTS)
      IAM = OMP_GET_THREAD_NUM()
      IPOINTS = NPOINTS/16
      CALL DO_BY_16(X,IAM,IPOINTS)
C$OMP END PARALLEL
      END

      SUBROUTINE DO_BY_16(X,IAM,IPOINTS)
      INTEGER X(*)
      I1 = IAM * IPOINTS + 1
      I2 = I1 + IPOINTS - 1
      DO 100 I=I1,I2
	X(I) = IAM
  100 CONTINUE
      END

      PROGRAM SNT001
      INTEGER I, ERRORS
      PARAMETER(N = 173)
      INTEGER X(N)
      DO 200 I=1,N
        X(I) = -1
  200 CONTINUE
      CALL SNT001S(X, N)
C       Determine last element modified
      ILAST = 0
      DO 300 I=1,N
        IF (X(I) .LT. 0) GOTO 400
	ILAST = I
  300 CONTINUE
  400 CONTINUE
      ERRORS = 0
C       Number of threads should be 16
      NT = X(ILAST)+1
      IF ( NT .NE. 16 ) THEN
        ERRORS = ERRORS + 1
        PRINT *,'stnthd001 - Expected 16 threads, observed ', NT
      ENDIF
C       Should be 160 points modified
      IF ( ILAST .NE. 160 ) THEN
        ERRORS = ERRORS + 1
        PRINT *,'stnthd001 - Wrong number of points modified'
      ENDIF
      IF (ERRORS .EQ. 0) THEN
        WRITE (*,'(A)') 'stnthd001 PASSED'
      ELSE
        WRITE (*,'(A)') 'stnthd001 FAILED'
      ENDIF
      END
