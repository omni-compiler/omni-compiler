C ********************************************************************
C                OpenMP Fortran API Test Suite
C                -----------------------------
C
C Test Name    : parsec001
C
C Summary      : Simple example of parallel sections
C
C Description  : F77 version of example A.8 from specification.
C
C Verification : Execution self-checks verify results but not work
C                sharing.
C
C Origin       : Example A.8 from OpenMP Fortran API specification
C
C Keywords     : F77, parallel sections
C
C Source Form  : Fixed
C
C Last Changed : $Date: 2004/02/06 18:15:44 $
C
C ********************************************************************

      SUBROUTINE PS001S()
C$OMP PARALLEL SECTIONS
C$OMP SECTION
      CALL XAXIS
C$OMP SECTION
      CALL YAXIS
C$OMP SECTION
      CALL ZAXIS
C$OMP END PARALLEL SECTIONS
      END

      SUBROUTINE XAXIS
      COMMON //AXIS
      INTEGER AXIS(3)
      AXIS(1) = 1
      CALL MSG("XAXIS",1)
      CALL MSG("XAXIS",2)
      CALL MSG("XAXIS",3)
      END

      SUBROUTINE YAXIS
      COMMON //AXIS
      INTEGER AXIS(3)
      AXIS(2) = 1
      CALL MSG("YAXIS",1)
      CALL MSG("YAXIS",2)
      CALL MSG("YAXIS",3)
      END

      SUBROUTINE ZAXIS
      COMMON //AXIS
      INTEGER AXIS(3)
      AXIS(3) = 1
      CALL MSG("ZAXIS",1)
      CALL MSG("ZAXIS",2)
      CALL MSG("ZAXIS",3)
      END

      SUBROUTINE MSG(WHO,COUNT)
c      CHARACTER(LEN=5) WHO
      CHARACTER*5 WHO
      INTEGER COUNT
      PRINT *,WHO,': MESSAGE ',COUNT
      END

      PROGRAM PS001
      COMMON //AXIS
      INTEGER AXIS(3)
      INTEGER I, ERRORS
C       Element 1 is the index of the next item to be served
      CALL PS001S()
C       Check elements of queue
      ERRORS = 0
      DO 100 I=1,3
        IF (AXIS(I) .NE. 1) THEN
          ERRORS = ERRORS + 1
          PRINT *,'parsec001 - expected AXIS(', I, ') = 1',
     1            ', observed', AXIS(I)
        ENDIF
  100 CONTINUE
      IF (ERRORS .EQ. 0) THEN
        WRITE (*,'(A)') 'parsec001 PASSED'
      ELSE
        WRITE (*,'(A)') 'parsec001 FAILED'
      ENDIF
      END
