C ********************************************************************
C                OpenMP Fortran API Test Suite
C                -----------------------------
C
C Test Name    : sngl001
C
C Summary      : Simple example of single construct
C
C Description  : F77 version of example A.9 from specification.
C
C Verification : Execution self-checks verify results but not work
C                sharing.
C
C Origin       : Example A.9 from OpenMP Fortran API specification
C
C Keywords     : F77, parallel, single, barrier, default
C
C Source Form  : Fixed
C
C Last Changed : $Date: 2004/02/06 18:15:44 $
C
C ********************************************************************

      SUBROUTINE SNGL001S(X, Y)
      INTEGER X, Y
C$OMP PARALLEL DEFAULT(SHARED)
      CALL WORK(X)
C$OMP BARRIER
C$OMP SINGLE
      CALL OUTPUT(X)
      CALL INPUT(Y)
C$OMP END SINGLE
      CALL WORK(Y)
C$OMP END PARALLEL
      END

      SUBROUTINE WORK(X)
      INTEGER X
C$OMP ATOMIC
      X = X + 1
      END

      SUBROUTINE OUTPUT(X)
      INTEGER X
      X = X + 10
      END

      SUBROUTINE INPUT(Y)
      INTEGER Y
      Y = Y + 100
      END

      PROGRAM SNGL001
      INTEGER I, ERRORS
      INTEGER X, Y
      X = 0
      Y = -100
      CALL SNGL001S(X, Y)
      ERRORS = 0
      IF ( X .LE. 0 ) THEN
        ERRORS = ERRORS + 1
        PRINT *,'sngl001 - Expect positive value, observe X = ', X
      ENDIF
      IF ( Y .LE. 0 ) THEN
        ERRORS = ERRORS + 1
        PRINT *,'sngl001 - Expect positive value, observe Y = ', Y
      ELSE
        PRINT *,'sngl001 - Apparent number of threads, Y = ', Y
      ENDIF
      IF ( X-Y .NE. 10 ) THEN
        ERRORS = ERRORS + 1
        PRINT *,'sngl001 - Expect difference of 10, observe X = ', X,
     1          ', Y = ', Y
      ENDIF
      IF (ERRORS .EQ. 0) THEN
        WRITE (*,'(A)') 'sngl001 PASSED'
      ELSE
        WRITE (*,'(A)') 'sngl001 FAILED'
      ENDIF
      END
