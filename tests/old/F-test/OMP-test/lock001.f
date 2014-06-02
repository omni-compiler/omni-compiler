C ********************************************************************
C                OpenMP Fortran API Test Suite
C                -----------------------------
C
C Test Name    : lock001
C
C Summary      : Simple example of using locks
C
C Description  : F77 version of example A.15 from specification.
C
C Verification : Execution self-checks verify results but not work
C                sharing.
C
C Origin       : Example A.15 from OpenMP Fortran API specification
C
C Keywords     : F77, parallel, shared, private, omp_get_thread_num
C Keywords     : omp_set_lock, omp_unset_lock, omp_test_lock
C
C Source Form  : Fixed
C
C Last Changed : $Date: 2004/02/06 18:15:44 $
C
C ********************************************************************

      SUBROUTINE LOCK_USAGE
      INTEGER   OMP_GET_THREAD_NUM
      EXTERNAL  OMP_GET_THREAD_NUM
      LOGICAL   OMP_TEST_LOCK
      EXTERNAL  OMP_TEST_LOCK

      INTEGER   LCK   ! THIS VARIABLE SHOULD BE POINTER SIZED

      CALL OMP_INIT_LOCK(LCK)
C$OMP PARALLEL SHARED(LCK) PRIVATE(ID)
      ID = OMP_GET_THREAD_NUM()
      CALL OMP_SET_LOCK(LCK)
      PRINT *, 'MY THREAD ID IS ', ID
      CALL OMP_UNSET_LOCK(LCK)

  100 CONTINUE
      IF (.NOT. OMP_TEST_LOCK(LCK)) THEN
C       WE DO NOT YET HAVE THE LOCK
C       SO WE MUST DO SOMETHING ELSE
        CALL SKIP(ID)
        GO TO 100
      ENDIF
C     WE NOW HAVE THE LOCK, AND CAN DO THE WORK
      CALL WORK(ID)
      CALL OMP_UNSET_LOCK(LCK)
C$OMP END PARALLEL

      CALL OMP_DESTROY_LOCK(LCK)

      END

      SUBROUTINE SKIP(I)
      PARAMETER(N=1024)
      COMMON /S/IS(N)
      IF ( I .LT. N ) THEN
        IS(I+1) = IS(I+1) + 1
      ENDIF
      END

      SUBROUTINE WORK(I)
      PARAMETER(N=1024)
      COMMON /W/NW,IW(N)
      CHARACTER*32 A
      PRINT *, 'lock001 - Doing work for thread ', I
      IF ( I .LT. N ) THEN
        IW(I+1) = I+1
        NW = NW + 1
      ENDIF
      END

      PROGRAM LOCK001
      PARAMETER(N=1024)
      COMMON /S/IS(N)
      COMMON /W/NW,IW(N)
      INTEGER I, ERRORS
      NW = 0
      DO 200 I=1,N
        IS(I) = 0
        IW(I) = -1
  200 CONTINUE
      CALL LOCK_USAGE()
C       Determine last element modified
      NT = 0
      DO 300 I=1,N
        IF (IW(I) .LT. 0) GOTO 400
        NT = I
  300 CONTINUE
  400 CONTINUE
      ERRORS = 0
      DO 500 I=1,NT
        IF (IW(I) .NE. I) THEN
          ERRORS = ERRORS + 1
          PRINT *, 'lock001 - EXPECTED IW(', I, ') = ', I,
     1                     ', OBSERVED ', IW(I)
        ENDIF
  500 CONTINUE
      IF ( NW .NE. NT ) THEN
        ERRORS = ERRORS + 1
        PRINT *, 'lock001 - EXPECTED NW = ', NT, ', OBSERVED ', NW
      ENDIF
      DO 600 I=1,NT
        IF ( IS(I) .GT. 0 ) THEN
          PRINT *, 'lock001 - thread ', I-1, ' skip count = ', IS(I)
        ENDIF
  600 CONTINUE
      PRINT *, 'lock001 - Apparent number of threads is ', NT
      IF (ERRORS .EQ. 0) THEN
        WRITE (*,'(A)') 'lock001 PASSED'
      ELSE
        WRITE (*,'(A)') 'lock001 FAILED'
      ENDIF
      END
