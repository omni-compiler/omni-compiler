C ********************************************************************
C                OpenMP Fortran API Test Suite
C                -----------------------------
C
C Test Name    : pardo001
C
C Summary      : Simple example of parallel do construct
C
C Description  : F77 version of example A.1 from specification.
C
C Verification : Execution self-checks verify results but not work
C                sharing.
C
C Origin       : Example A.1 from OpenMP Fortran API specification
C
C Keywords     : F77, parallel do
C
C Source Form  : Fixed
C
C Last Changed : $Date: 2004/02/06 18:15:44 $
C
C ********************************************************************

      SUBROUTINE PD001S(N,A,B)
      INTEGER N,I
      INTEGER A(N), B(N)
C$OMP PARALLEL DO
      DO 100 I=2,N
        B(I) = (A(I) + A(I-1)) / 2
  100 CONTINUE
C$OMP END PARALLEL DO
      END

      PROGRAM PD001
      INTEGER I, ERRORS
      PARAMETER(N = 1024)
      INTEGER A(N), B(N)
      DO 200 I=1,N
        A(I) = 2*I + 1
        B(I) = 0
  200 CONTINUE
      CALL PD001S(N, A, B)
      ERRORS = 0
      DO 300 I=2,N
        IF (B(I) .NE. 2*I) THEN
          ERRORS = ERRORS + 1
	  IF (ERRORS .EQ. 1) THEN
            PRINT *,'pardo001 - VALUES IN B ARE NOT AS EXPECTED'
	  ENDIF
          PRINT *,'EXPECTED B(', I, ') = ', 2*I, ' OBSERVED ', B(I)
        ENDIF
  300 CONTINUE
      IF (ERRORS .EQ. 0) THEN
        WRITE (*,'(A)') 'pardo001 PASSED'
      ELSE
        WRITE (*,'(A)') 'pardo001 FAILED'
      ENDIF
      END
