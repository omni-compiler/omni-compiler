C ********************************************************************
C                OpenMP Fortran API Test Suite
C                -----------------------------
C
C Test Name    : do001
C
C Summary      : do construct with nowait clause
C
C Description  : F77 version of example A.4 from specification.
C
C Verification : Execution self-checks verify results but not work
C                sharing.
C
C Origin       : Example A.4 from OpenMP Fortran API specification
C
C Keywords     : F77, do, nowait
C
C Source Form  : Fixed
C
C Last Changed : $Date: 2004/02/06 18:15:44 $
C
C ********************************************************************

      SUBROUTINE D001S(M,N,A,B,Y,Z)
      INTEGER M,N,I
      INTEGER A(N), B(N), Y(M), Z(M), SQRT
      EXTERNAL SQRT
C$OMP PARALLEL
C$OMP DO
      DO 100 I=2,N
        B(I) = (A(I) + A(I-1)) / 2
  100 CONTINUE
C$OMP END DO NOWAIT
C$OMP DO
      DO 200 I=1,M
        Y(I) = SQRT(Z(I))
  200 CONTINUE
C$OMP END DO NOWAIT
C$OMP END PARALLEL
      END

      INTEGER FUNCTION SQRT(K)
      J = 1
      DO 250 I=1,K
      IF ( (J*J) .NE. K ) THEN
	J = (J + (K/J))/2
      ELSE
	GOTO 260
      ENDIF
  250 CONTINUE
  260 SQRT = J
      END

      PROGRAM D001
      INTEGER M, N, I, ERRORS
      PARAMETER(M = 117)
      PARAMETER(N = 511)
      INTEGER A(N), B(N), Y(M), Z(M)
      DO 300 I=1,N
        A(I) = 2*I + 1
        B(I) = 0
  300 CONTINUE
      DO 400 I=1,M
        Z(I) = I*I
        Y(I) = 0
  400 CONTINUE
      CALL D001S(M, N, A, B, Y, Z)
      ERRORS = 0
      DO 500 I=2,N
        IF (B(I) .NE. 2*I) THEN
          ERRORS = ERRORS + 1
	  IF (ERRORS .EQ. 1) THEN
            write(6,*)'do001 - VALUES IN B ARE NOT AS EXPECTED'
	  ENDIF
          write(6,*)'EXPECTED B(', I, ') = ', 2*I, ' OBSERVED ', B(I)
        ENDIF
  500 CONTINUE
      DO 600 I=1,M
        IF (Y(I) .NE. I) THEN
          ERRORS = ERRORS + 1
	  IF (ERRORS .EQ. 1) THEN
            write(6,*)'do001 - VALUES IN Y ARE NOT AS EXPECTED'
	  ENDIF
          write(6,*) 'EXPECTED Y(', I, ') = ', I, ' OBSERVED ', Y(I)
        ENDIF
  600 CONTINUE
      IF (ERRORS .EQ. 0) THEN
        WRITE (*,'(A)') 'do001 PASSED'
      ELSE
        WRITE (*,'(A)') 'do001 FAILED'
      ENDIF
      END
