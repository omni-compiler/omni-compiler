      SUBROUTINE NOZZLE (TIME,P0,D0,E0,C0,PRES,DENS,VEL,ENER,FLOW,
     *     GAMMA)
C
C        DEFINE TIME DEPENDENT FLOW CONDITIONS AT THE NOZZLE EXIT
C
C
C        INPUT:
C
C        TIME      REAL      TIME (S) TO DEFINE CONDITIONS AT
C        P0        REAL      INITIAL CHAMBER PRESSURE (PASCALS)
C        D0        REAL      INITIAL CHAMBER DENSITY (KG/M**3)
C                              NOT REQUIRED IF FLOW NOT SET
C        E0        REAL      INITIAL CHAMBER SPECIFIC INTERNAL ENERGY
C                                  (J/KG)
C        C0        REAL      INITIAL CHAMBER SOUND VELOCITY (M/S)
C                              NOT REQUIRED IF FLOW NOT SET
C        FLOW      REAL      IF NON-NEGATIVE, THIS DEFINES THE MASS
C                              FLOW RATE THROUGH THE NOZZLE.  IF SET,
C                              FLOW WILL BE CONSTANT IN TIME.
C        GAMMA     REAL      INITIAL CHAMBER GAMMA (NOT REQUIRED IF
C                              FLOW IS GIVEN)
C
C        OUTPUT:
C
C        PRES      REAL      PRESSURE (PASCALS)
C        DENS      REAL      DENSITY (KG/M**3)
C        VEL       REAL      VELOCITY (M/S)
C        ENER      REAL      INTERNAL SPECIFIC ENERGY (J/KG)
C
C        IF FLOW IS NOT SET, THEN
C        WE ASSUME CRITICAL FLOW ACROSS THE NOZZLE AND CALCULATE THE
C        DECAY IN THE CHAMBER ASSUMING ISOTHERMAL CONDITIONS.
C
C
C        AREA = NOZZLE AREA (M**2)
C        VOL  = VOLUME OF CHAMBER (M**3)
C
      PARAMETER (AREA=5.1725E-7,VOL=1.24E-5)
      IMPLICIT REAL (A-H,O-Z)
      PARAMETER (IMAX=500)
      DIMENSION P(IMAX),T(IMAX)
      INTEGER TYPE
      LOGICAL FIRST,EOFF
      CHARACTER*80 CARD,FIELD
      CHARACTER*30 INFIL
      COMMON /CIMAGE/ CARD,FIELD
      COMMON /IMAGE/ EOFF,ICPNT
      DATA FIRST /.TRUE./
C
C        IF FLOW NOT SET, USE INPUTED PRESSURE-TIME HISTORY
C
      IF(FLOW.LE.0.0) THEN
          IF(FIRST) THEN
              ICPNT=999
              ICNT=1
              WRITE(*,'(//'' ENTER P-T HISTORY FILE NAME: '')')
              READ(*,'(A)') INFIL
              OPEN(UNIT=5,FILE=INFIL,STATUS='UNKNOWN')
              REWIND 5
3             CALL VALUE(T(ICNT),TYPE)
              IF(.NOT.EOFF) THEN
                  IF(TYPE.NE.1) THEN
                      CALL TERROR(1,FIELD,TYPE)
                      WRITE(6,'('' ABORT IN NOZZLE'')')
                      STOP
                  ENDIF
                  CALL VALUE(P(ICNT),TYPE)
                  IF(EOFF) THEN
                      WRITE(6,'('' UNEXPECTED EOF IN NOZZLE, ABORT'')')
                      STOP
                  ELSE IF(TYPE.NE.1) THEN
                      CALL TERROR(1,FIELD,TYPE)
                      WRITE(6,'('' ABORT IN NOZZLE'')')
                      STOP
                  ENDIF
                  ICNT=ICNT+1
                  GO TO 3
              ELSE
                  ICNT=ICNT-1
                  IF(ICNT.LE.0) THEN
                      WRITE(6,'('' NO NOZZLE P/T HISTORY, ABORT'')')
                      CLOSE(UNIT=5)
                      STOP
                  ENDIF
                  CLOSE(UNIT=5)
                  WRITE(6,'(''1PRESSURE-TIME HISTORY FOR CHAMBER''/)')
                  DO 99 IJ=1,ICNT
                      P(IJ)=P(IJ)/1.727865
                      WRITE(6,'('' TIME='',1PE12.5,'' PRES='',E12.5)')
     *                    T(IJ),P(IJ)
  99              CONTINUE
                  P0=P(1)
                  FIRST=.FALSE.
                  IOLD=ICNT-1
                  RETURN
              ENDIF
          ENDIF
          IF(TIME.GE.T(ICNT)) THEN
              N=ICNT-1
          ELSEIF (TIME.LT.T(1)) THEN
              N=1
          ELSE
              NN=ICNT-1
              IF(TIME.LT.T(IOLD+1)) NN=IOLD
              DO 10 I=NN,1,-1
                  N=I
                  IF(TIME.GE.T(I).AND.TIME.LT.T(I+1)) GO TO 20
   10         CONTINUE
          ENDIF
   20     X1=T(N)
          Y1=P(N)
          X2=T(N+1)
          Y2=T(N+2)
          IOLD=N
          SLOPE=(Y2-Y1)/(X2-X1)
          B=Y2-SLOPE*X2
          PRES=TIME*SLOPE+B
          ENER=E0
          DENS=PRES/((GAMMA-1.0)*ENER)
          VEL=C0
      ELSE
C
C        FLOW IS SET, CALCULATE CONSTANT FLOW
C
          VEL=FLOW/(D0*AREA)
          ENER=E0
          DENS=D0
          PRES=P0
      ENDIF
      END
