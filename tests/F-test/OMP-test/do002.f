C ********************************************************************
C                OpenMP Fortran API Test Suite
C                -----------------------------
C
C Test Name    : do002
C
C Summary      : Simple example of lastprivate construct
C
C Description  : F77 version of example A.6 from specification.
C
C Verification : Execution self-checks verify results but not work
C                sharing.
C
C Origin       : Example A.6 from OpenMP Fortran API specification
C
C Keywords     : F77, parallel, lastprivate
C
C Source Form  : Fixed
C
C Last Changed : $Date: 2004/02/06 18:15:44 $
C
C ********************************************************************

      SUBROUTINE D002S(N,A,B,C)
      INTEGER N, A(N), B(N), C(N), I
C$OMP PARALLEL
C$OMP DO LASTPRIVATE(I)
      DO 100 I=1,N
        A(I) = B(I) + C(I)
  100 CONTINUE
C$OMP END PARALLEL
      CALL REVERSE(I)
      END

      SUBROUTINE REVERSE(I)
      INTEGER I, J
      COMMON // J
      J = I
      END

      PROGRAM D002
      INTEGER I, J, ERRORS
      COMMON // J
      PARAMETER(N = 31)
      INTEGER A(N), B(N), C(N)
      DO 200 I=1,N
        A(I) = 0
        B(I) = I
        C(I) = 2*I
  200 CONTINUE
      CALL D002S(N, A, B, C)
      ERRORS = 0
      IF (J .NE. N+1) THEN
      ERRORS = ERRORS + 1
        write(6,*)'do002 - EXPECTED J = ', N+1, ' OBSERVED ', J
      ENDIF
      DO 300 I=1,N
        IF (A(I) .NE. 3*I) THEN
          ERRORS = ERRORS + 1
        IF (ERRORS .EQ. 1) THEN
            write(6,*)'do002 - VALUES IN A ARE NOT AS EXPECTED'
        ENDIF
          write(6,*) 'EXPECTED A(', I, ') = ', 3*I, ' OBSERVED ', A(I)
        ENDIF
  300 CONTINUE
      IF (ERRORS .EQ. 0) THEN
        WRITE (*,'(A)') 'do002 PASSED'
      ELSE
        WRITE (*,'(A)') 'do002 FAILED'
      ENDIF
      END
