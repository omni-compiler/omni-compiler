C ********************************************************************
C                OpenMP Fortran API Test Suite
C                -----------------------------
C
C Test Name    : do003
C
C Summary      : Two examples of nested do directives
C
C Description  : F77 version of example A.16 from specification.
C
C Verification : Execution self-checks verify results but not work
C                sharing.
C
C Origin       : Example A.16 from OpenMP Fortran API specification
C
C Keywords     : F77, parallel, do, default, shared
C
C Source Form  : Fixed
C
C Last Changed : $Date: 2004/02/06 18:15:44 $
C
C ********************************************************************

      SUBROUTINE D003S(N)
C$OMP PARALLEL DEFAULT(SHARED)
C$OMP DO
      DO 200 I=1, N
C$OMP PARALLEL SHARED(I, N)
C$OMP DO
        DO 100 J=1, N
          CALL WORK(I, J)
  100   CONTINUE
C$OMP END PARALLEL
  200 CONTINUE
C$OMP END PARALLEL
      END

      SUBROUTINE D003T(N)
C$OMP PARALLEL DEFAULT(SHARED)
C$OMP DO
      DO 300 I=1, N
	CALL SOME_WORK(I, N)
  300 CONTINUE
C$OMP END PARALLEL
      END

      SUBROUTINE SOME_WORK(I, N)
C$OMP PARALLEL DEFAULT(SHARED)
C$OMP DO
      DO 400 J=1, N
        CALL WORK(I, J)
  400 CONTINUE
C$OMP END PARALLEL
      RETURN
      END

      SUBROUTINE WORK(I,J)
      PARAMETER(N=117)
      COMMON /W/IW(N,N)
      IW(I,J) = I + J*N
      END

      SUBROUTINE INIT()
      PARAMETER(N=117)
      COMMON /W/IW(N,N)
      DO 600 I=1,N
        DO 500 J=1,N
          IW(I,J) = 0
  500   CONTINUE
  600 CONTINUE
      END

      FUNCTION ICHECK(MSG)
      CHARACTER*(*) MSG
      PARAMETER(N=117)
      COMMON /W/IW(N,N)
      ICHECK = 0
      DO 800 I=1,N
        DO 700 J=1,N
	  IEXPECT = I + J*N
          IF ( IW(I,J) .NE. IEXPECT ) THEN
	    ICHECK = ICHECK + 1
            WRITE(6,*) 'do003 - EXPECTED IW(', I, ',', J, ') = ', 
     1           IEXPECT, ', OBSERVED ', IW(I,J), MSG
	  ENDIF
  700   CONTINUE
  800 CONTINUE
      END

      PROGRAM D003
      INTEGER I, ERRORS
      PARAMETER(N = 117)
      COMMON /W/IW(N,N)
      ERRORS = 0
      CALL INIT()
      CALL D003S(N)
      ERRORS = ERRORS + ICHECK('after D003S')
      CALL INIT()
      CALL D003T(N)
      ERRORS = ERRORS + ICHECK('after D003T')
      IF (ERRORS .EQ. 0) THEN
        WRITE (*,'(A)') 'do003 PASSED'
      ELSE
        WRITE (*,'(A)') 'do003 FAILED'
      ENDIF
      END
