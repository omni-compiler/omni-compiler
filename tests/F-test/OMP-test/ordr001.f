C ********************************************************************
C                OpenMP Fortran API Test Suite
C                -----------------------------
C
C Test Name    : ordr001
C
C Summary      : Simple example of ordered construct
C
C Description  : F77 version of example A.10 from specification.
C
C Verification : Execution self-checks verify results but not work
C                sharing.
C
C Origin       : Example A.10 from OpenMP Fortran API specification
C
C Keywords     : F77, parallel, ordered, schedule, dynamic
C
C Source Form  : Fixed
C
C Last Changed : $Date: 2004/02/06 18:15:44 $
C
C ********************************************************************

      SUBROUTINE RDR002S(LB,UB,ST)
      INTEGER LB, UB, ST
C$OMP PARALLEL
!C$OMP DO ORDERED SCHEDULE(DYNAMIC)
C$OMP DO ORDERED SCHEDULE(STATIC)
      DO 100 I=LB,UB,ST
        CALL WORK(I)
  100 CONTINUE
C$OMP END PARALLEL
      END

      SUBROUTINE WORK(K)
      COMMON //IAN,IA(317)
C$OMP ORDERED
      IAN = IAN + 1
      IA(IAN) = K
      WRITE (*,*) K
C$OMP END ORDERED
      END

      PROGRAM RDR002
      INTEGER I, ERRORS
      PARAMETER(N = 317)
      COMMON //IAN,IA(N)
      IAN = 0
      DO 200 I=1,N
        IA(I) = 0
  200 CONTINUE
      CALL RDR002S(1,N*2,2)
      ERRORS = 0
      IF (IAN .NE. N) THEN
	ERRORS = ERRORS + 1
        PRINT *,'ordr001 - EXPECTED IAN = ', N, ' OBSERVED ', IAN
      ENDIF
      DO 300 I=1,N
        IF (IA(I) .NE. 2*I-1) THEN
          ERRORS = ERRORS + 1
	  IF (ERRORS .EQ. 1) THEN
            PRINT *,'ordr001 - VALUES IN IA ARE NOT AS EXPECTED'
	  ENDIF
          PRINT *,'EXPECTED IA(', I, ') = ', 2*I-1, ' OBSERVED ', IA(I)
        ENDIF
  300 CONTINUE
      IF (ERRORS .EQ. 0) THEN
        WRITE (*,'(A)') 'ordr001 PASSED'
      ELSE
        WRITE (*,'(A)') 'ordr001 FAILED'
      ENDIF
      END
