C ********************************************************************
C                OpenMP Fortran API Test Suite
C                -----------------------------
C
C Test Name    : par002
C
C Summary      : Example of private and firstprivate scoping
C
C Description  : F77 version of example A.19 from specification.
C
C Verification : Execution self-checks verify results but not work
C                sharing.
C
C Origin       : Example A.19 from OpenMP Fortran API specification
C
C Keywords     : F77, parallel, private, firstprivate
C
C Source Form  : Fixed
C
C Last Changed : $Date: 2004/02/06 18:15:44 $
C
C ********************************************************************

      SUBROUTINE PAR002S(I, J, K)
      INTEGER I, J
      I = 1
      J = 2
      K = 0
C$OMP PARALLEL PRIVATE(I) FIRSTPRIVATE(J) SHARED(K)
      I = 3
      J = J + 2
C$OMP ATOMIC
      K = K + 1
C$OMP END PARALLEL
      PRINT *, 'par002 - values of I, J, and K: ', I, J, K
      END

      PROGRAM PAR002
      INTEGER I, J, K
      CALL PAR002S(I, J, K)
      IF ( K .LE. 1 ) THEN
	PRINT *,'par002 - NOTE serial or team size of one'
      ENDIF
      IF ( I .EQ. 1 .AND. J .EQ. 2 ) THEN
	PRINT *,'par002 - NOTE original variable retains original value'
      ENDIF
      IF ( I .EQ. 3 ) THEN
        PRINT *,'par002 - NOTE original variable gets value of master',
     1	        ' copy of private variable'
        IF ( J .EQ. 4 ) THEN
          PRINT *,'par002 - NOTE value of J ',
     1            'gives no evidence of parallel execution'
        ENDIF
      ENDIF
      WRITE (*,'(A)') 'par002 PASSED'
      END
