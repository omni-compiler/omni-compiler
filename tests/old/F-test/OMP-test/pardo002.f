C ********************************************************************
C                OpenMP Fortran API Test Suite
C                -----------------------------
C
C Test Name    : pardo002
C
C Summary      : Parallel do construct with default and reduction
C
C Description  : F77 version of example A.7 from specification.
C
C Verification : Execution self-checks verify results but not work
C                sharing.
C
C Origin       : Example A.7 from OpenMP Fortran API specification
C
C Keywords     : F77, parallel do, default, reduction
C
C Source Form  : Fixed
C
C Last Changed : $Date: 2004/02/06 18:15:44 $
C
C ********************************************************************

      SUBROUTINE PD002S(N,A,B)
      INTEGER N,A,B,I,ALOCAL,BLOCAL
C$OMP PARALLEL DEFAULT(PRIVATE) 
C$OMP DO REDUCTION(+: A,B) FIRSTPRIVATE(N)
      DO 100 I=1,N
        CALL WORK(ALOCAL,BLOCAL)
        A = A + ALOCAL
        B = B + BLOCAL
  100 CONTINUE
C$OMP END PARALLEL
      END

      SUBROUTINE WORK(I,J)
      INTEGER I,J
      I = 1
      J = -1
      END

      PROGRAM PD002
      INTEGER N, I, ERRORS
      PARAMETER(N = 1024)
      INTEGER A, B
      A = 0
      B = N
      CALL PD002S(N, A, B)
      ERRORS = 0
      IF (A .NE. N) THEN
        ERRORS = ERRORS + 1
        PRINT *,'pardo002 - EXPECTED A = ', N, ' OBSERVED ', A
      ENDIF
      IF (B .NE. 0) THEN
        ERRORS = ERRORS + 1
        PRINT *,'pardo002 - EXPECTED B = ', 0, ' OBSERVED ', B
      ENDIF
      IF (ERRORS .EQ. 0) THEN
        WRITE (*,'(A)') 'pardo002 PASSED'
      ELSE
        WRITE (*,'(A)') 'pardo002 FAILED'
      ENDIF
      END
