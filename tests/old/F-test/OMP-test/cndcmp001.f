C ********************************************************************
C                OpenMP Fortran API Test Suite
C                -----------------------------
C
C Test Name    : cndcmp001
C
C Summary      : Simple example of conditional compilation
C
C Description  : F77 version of example A.2 from specification.
C
C Verification : Execution self-checks verify results but not work
C                sharing.
C
C Origin       : Example A.2 from OpenMP Fortran API specification
C
C Keywords     : F77, conditional compilation
C
C Source Form  : Fixed
C
C Last Changed : $Date: 2004/02/06 18:15:44 $
C
C ********************************************************************

      SUBROUTINE CC001S(N,A,B)
      INTEGER N,I
      INTEGER A(N), B(N)
C$OMP PARALLEL DO
C$    DO 100 I=2,N
!$      B(I) = (A(I) + A(I-1)) / 2
*$100 CONTINUE
C$OMP END PARALLEL DO
      END

      PROGRAM CC001
      INTEGER I, ERRORS
      PARAMETER(N = 1003)
      PARAMETER(MAXMESSAGES = 10)
      INTEGER A(N), B(N)
      DO 200 I=1,N
        A(I) = 2*I + 1
        B(I) = 0
  200 CONTINUE
      CALL CC001S(N, A, B)
      ERRORS = 0
      DO 300 I=2,N
        IF (B(I) .NE. 2*I) THEN
          ERRORS = ERRORS + 1
          IF (ERRORS .EQ. 1) THEN
            PRINT *,'cndcmp001 - VALUES IN B ARE NOT AS EXPECTED'
          ENDIF
	  IF (ERRORS .LE. MAXMESSAGES) THEN
            PRINT *,'EXPECTED B(', I, ') = ', 2*I, ' OBSERVED ', B(I)
	  ENDIF
        ENDIF
  300 CONTINUE
      IF (ERRORS .GT. MAXMESSAGES) THEN
	PRINT *,'... (', ERRORS - MAXMESSAGES,
     1          ' additional messages not shown)'
      ENDIF
      IF (ERRORS .EQ. 0) THEN
        WRITE (*,'(A)') 'cndcmp001 PASSED'
      ELSE
        WRITE (*,'(A)') 'cndcmp001 FAILED'
      ENDIF
      END
