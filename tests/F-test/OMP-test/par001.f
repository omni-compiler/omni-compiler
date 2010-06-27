C ********************************************************************
C                OpenMP Fortran API Test Suite
C                -----------------------------
C
C Test Name    : par001
C
C Summary      : Simple example of parallel construct
C
C Description  : F77 version of example A.3 from specification.
C
C Verification : Execution self-checks verify results but not work
C                sharing.
C
C Origin       : Example A.3 from OpenMP Fortran API specification
C
C Keywords     : F77, parallel
C
C Source Form  : Fixed
C
C Last Changed : $Date: 2004/02/06 18:15:44 $
C
C ********************************************************************

      SUBROUTINE PAR001S(X,NPOINTS)
      INTEGER OMP_GET_NUM_THREADS, OMP_GET_THREAD_NUM
      EXTERNAL OMP_GET_NUM_THREADS, OMP_GET_THREAD_NUM
      INTEGER X(NPOINTS)
C$OMP PARALLEL DEFAULT(PRIVATE) SHARED(X,NPOINTS)
      IAM = 0
      NP = 1
C$      IAM = OMP_GET_THREAD_NUM()
C$      NP =  OMP_GET_NUM_THREADS()
      IPOINTS = NPOINTS/NP
      CALL SUBDOMAIN(X,IAM,IPOINTS)
C$OMP END PARALLEL
      END

      SUBROUTINE SUBDOMAIN(X,IAM,IPOINTS)
      INTEGER X(*)
      I1 = IAM * IPOINTS + 1
      I2 = I1 + IPOINTS - 1
      DO 100 I=I1,I2
	X(I) = IAM
  100 CONTINUE
      END

      PROGRAM PAR001
      INTEGER I, ERRORS
      PARAMETER(N = 1024)
      INTEGER X(N)
      DO 200 I=1,N
        X(I) = -1
  200 CONTINUE
      CALL PAR001S(X, N)
C       Determine last element modified
      ILAST = 0
      DO 300 I=1,N
        IF (X(I) .LT. 0) GOTO 400
	ILAST = I
  300 CONTINUE
  400 CONTINUE
C       Infer number of threads
      NT = X(ILAST)+1
      ERRORS = 0
C       Should be fewer than NT points not modified
      IF ( N-ILAST .GE. NT ) THEN
        ERRORS = ERRORS + 1
        PRINT *,'par001 - Too few points changed for static dist'
      ENDIF
C       Number of threads should evenly divide points changed
      IF ( MOD(ILAST,NT) .NE. 0 ) THEN
        ERRORS = ERRORS + 1
        PRINT *,'par001 - Threads do not divide points changed'
      ENDIF
      PRINT *,'par001 - Apparent number threads = ', NT
      IF (ERRORS .EQ. 0) THEN
        WRITE (*,'(A)') 'par001 PASSED'
      ELSE
	PRINT *,'par001 -   Number points =', N
	PRINT *,'par001 -   Points changed =', ILAST
        WRITE (*,'(A)') 'par001 FAILED'
      ENDIF
      END
