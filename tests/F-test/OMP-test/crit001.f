C ********************************************************************
C                OpenMP Fortran API Test Suite
C                -----------------------------
C
C Test Name    : crit001
C
C Summary      : Critical constructs with different names
C
C Description  : F77 version of example A.5 from specification.
C
C Verification : Execution self-checks verify results but not work
C                sharing.
C
C Origin       : Example A.5 from OpenMP Fortran API specification
C
C Keywords     : F77, parallel, critical
C
C Source Form  : Fixed
C
C Last Changed : $Date: 2004/02/06 18:15:44 $
C
C ********************************************************************

      SUBROUTINE CRIT001S(N,X,Y)
      INTEGER N, X(N), Y(N), IX_NEXT, IY_NEXT
C$OMP PARALLEL DEFAULT(PRIVATE) SHARED(X,Y)
C$OMP CRITICAL(XAXIS)
      CALL DEQUEUE(IX_NEXT, X)
C$OMP END CRITICAL(XAXIS)
      CALL WORK(IX_NEXT, X)
C$OMP CRITICAL(YAXIS)
      CALL DEQUEUE(IY_NEXT, Y)
C$OMP END CRITICAL(YAXIS)
      CALL WORK(IY_NEXT, Y)
C$OMP END PARALLEL
      END

      SUBROUTINE DEQUEUE(NEXT,A)
      INTEGER A(*)
      NEXT = A(1)
      A(1) = A(1) + 1
      END

      SUBROUTINE WORK(I,A)
      INTEGER A(*)
      A(I) = - A(I) + I
      END

      PROGRAM CRIT001
      INTEGER N, I, ERRORS
      PARAMETER(N = 743)
      INTEGER X(N), Y(N)
      DO 200 I=1,N
        X(I) = -1
        Y(I) = -I
  200 CONTINUE
C       Element 1 is the index of the next item to be served
      X(1) = 2
      Y(1) = 2
      CALL CRIT001S(N,X,Y)
C       Check elements of queue
      ERRORS = 0
      DO 300 I=2,N
        J = I
        IF (X(I) .LT. 0) GOTO 400
        IF (X(I) .NE. I+1) THEN
          PRINT *,'crit001 - expected X(', I, ') =', I+1,
     1            ', observed', X(I)
        ENDIF
        IF (Y(I) .LT. 0) GOTO 400
        IF (Y(I) .NE. I+I) THEN
          PRINT *,'crit001 - expected Y(', I, ') =', I+I,
     1            ', observed', Y(I)
        ENDIF
  300 CONTINUE
  400 CONTINUE
      IF (X(J) .GE. 0 .OR. Y(J) .GE. 0) THEN
        PRINT *,'crit001 - inconsistent service on queues'
        PRINT *,'crit001 - expected X(', J, ') =', -1,
     1          ', observed', X(J)
        PRINT *,'crit001 - expected Y(', J, ') =', -J,
     1          ', observed', X(J)
      ENDIF
      PRINT *,'crit001 - apparently used', J-2, 'threads'
      IF (ERRORS .EQ. 0) THEN
        WRITE (*,'(A)') 'crit001 PASSED'
      ELSE
        WRITE (*,'(A)') 'crit001 FAILED'
      ENDIF
      END
