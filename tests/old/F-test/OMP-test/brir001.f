C ********************************************************************
C                OpenMP Fortran API Test Suite
C                -----------------------------
C
C Test Name    : brir001
C
C Summary      : Example of binding of barrier directives
C
C Description  : F77 version of example A.18 from specification.
C
C Verification : Execution self-checks verify results but not work
C                sharing.
C
C Origin       : Example A.18 from OpenMP Fortran API specification
C
C Keywords     : F77, parallel, do, private, shared, barrier
C
C Source Form  : Fixed
C
C Last Changed : $Date: 2004/02/06 18:15:44 $
C
C ********************************************************************

      SUBROUTINE SUB1(N)
C$OMP PARALLEL PRIVATE(I) SHARED(N)
C$OMP DO
      DO 100 I=1, N
        CALL SUB2(I)
  100   CONTINUE
C$OMP END PARALLEL
      END

      SUBROUTINE SUB2(K)
C$OMP PARALLEL SHARED(K)
      CALL SUB3(K)
C$OMP END PARALLEL
      END

      SUBROUTINE SUB3(N)
      CALL WORK(N,1)
C$OMP BARRIER
      CALL WORK(N,2)
      END

      SUBROUTINE WORK(I,J)
      PARAMETER(N=15)
      COMMON /W/NT,IW(N,2)
      IF ( J .EQ. 1 ) THEN
C$OMP   ATOMIC
        IW(N+1-I,J) = IW(N+1-I,J) + I
      ELSE
C$OMP   ATOMIC
        NT = NT + 1
        IW(I,J) = IW(I,J-1)
      ENDIF
      END

      SUBROUTINE INIT()
      PARAMETER(N=15)
      COMMON /W/NT,IW(N,2)
      NT = 0
      DO 600 J=1,2
        DO 500 I=1,N
          IW(I,J) = 0
  500   CONTINUE
  600 CONTINUE
      END

      FUNCTION ICHECK(K)
      PARAMETER(N=15)
      COMMON /W/NT,IW(N,2)
      ICHECK = 0
      DO 800 I=1,N
        DO 700 J=1,2
          IF ( IW(I,J) .GT. 0 ) THEN
            PRINT *,'brir001 - IW(', I, ',', J, ') = ', IW(I,J), 
     1                         ' AFTER SUB', K
          ENDIF
	  IF ( K .EQ. 1 ) THEN
C           LOOK FOR A RESULT THAT COULD NOT HAPPEN SERIALLY
	    IF ( J .EQ. 2 .AND. 
     1  	 ((IW(I,J) .GT. 0) .NEQV. (I .GE. (N+1)/2)) ) THEN
	      ICHECK = ICHECK + 1
	    ENDIF
	  ELSE
	    IF ( I .EQ. (N+1)/2 ) THEN
		IF ( IW(I,J) .NE. (I*NT) ) ICHECK = ICHECK + 1
	    ELSE
		IF ( IW(I,J) .NE. 0 ) ICHECK = ICHECK + 1
	    ENDIF
          ENDIF
  700   CONTINUE
  800 CONTINUE
      IF ( K .EQ. 1 ) THEN
	IF ( ICHECK .EQ. 0 ) THEN
	  PRINT *, 'brir001 - NOTE SUB1 ',
     1             'shows no evidence of parallel execution'
	ENDIF
        ICHECK = 0
      ELSE
        IF ( ICHECK .GT. 0 ) THEN
	  PRINT *, 'brir001 - ERROR SUB2 ',
     1             'synchronization was not effective'
	ENDIF
      ENDIF
      END

      PROGRAM B001
      INTEGER I, ERRORS
      PARAMETER(N=15)
      COMMON /W/NT,IW(N,2)
      ERRORS = 0
      CALL INIT()
      CALL SUB1(N)
      ERRORS = ERRORS + ICHECK(1)
      CALL INIT()
      CALL SUB2((N+1)/2)
      ERRORS = ERRORS + ICHECK(2)
      IF (ERRORS .EQ. 0) THEN
        WRITE (*,'(A)') 'brir001 PASSED'
      ELSE
        WRITE (*,'(A)') 'brir001 FAILED'
      ENDIF
      END
