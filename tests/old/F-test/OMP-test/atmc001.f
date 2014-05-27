C ********************************************************************
C                OpenMP Fortran API Test Suite
C                -----------------------------
C
C Test Name    : atmc001
C
C Summary      : Simple example of atomic directive
C
C Description  : F77 version of example A.12 from specification.
C
C Verification : Execution self-checks verify results but not work
C                sharing.
C
C Origin       : Example A.12 from OpenMP Fortran API specification
C
C Keywords     : F77, parallel do, atomic, default, shared
C
C Source Form  : Fixed
C
C Last Changed : $Date: 2004/02/06 18:15:44 $
C
C ********************************************************************

      SUBROUTINE AT001S(N,X,Y,INDEX)
      INTEGER N,I,XLOCAL,YLOCAL
      INTEGER X(2), Y(N), INDEX(N)
C$OMP PARALLEL DO DEFAULT(PRIVATE) SHARED(X, Y, INDEX, N)
      DO 100 I=1,N
	CALL WORK(XLOCAL, YLOCAL)
C$OMP ATOMIC
      X(INDEX(I)) = X(INDEX(I)) + XLOCAL
      Y(I) = Y(I) + YLOCAL
  100 CONTINUE
C$OMP END PARALLEL DO
      END

      SUBROUTINE WORK(I, J)
      I = 1
      J = 2
      END

      PROGRAM AT001
      INTEGER I, ERRORS, ERRORS1
      PARAMETER(N = 613)
      INTEGER X(2), Y(N), INDEX(N)
      X(1) = 0
      X(2) = 0
      DO 200 I=1,N
        Y(I) = I
	IF ( I .LE. N/2 ) THEN
	  INDEX(I) = 1
	ELSE
	  INDEX(I) = 2
	ENDIF
  200 CONTINUE
      CALL AT001S(N, X, Y, INDEX)
      ERRORS = 0
      IF ( X(1) .NE. N/2 ) THEN
	ERRORS = ERRORS + 1
	WRITE(6,*)'atmc001 - EXPECTED X(', I, ') = ', N/2,
     1                   ' OBSERVED ', X(I)
      ENDIF
      IF ( X(2) .NE. N - N/2 ) THEN
	ERRORS = ERRORS + 1
	WRITE(6,*)'atmc001 - EXPECTED X(', I, ') = ', N - N/2,
     1                   ' OBSERVED ', X(I)
      ENDIF
      DO 300 I=1,N
        IF (Y(I) .NE. 2+I) THEN
          WRITE(6,*)'atmc001 - EXPECTED Y(', I, ') = ', 2+I,
     1                     ' OBSERVED ', Y(I)
        ENDIF
  300 CONTINUE
      IF (ERRORS .EQ. 0) THEN
        WRITE (*,'(A)') 'atmc001 PASSED'
      ELSE
        WRITE (*,'(A)') 'atmc001 FAILED'
      ENDIF
      END
