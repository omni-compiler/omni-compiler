      SUBROUTINE READIN (PROB,TITLE,CSTOP,FCYCLE,DCYCLE,DHIST,VHIST,
     *                   IMAX,PHIST,DEBUG,NSTAT,STATS,MAXSTA,
     *                   NCORE,PPLOT,DPLOT,VPLOT,TPLOT,SLIST,D0,E0,
     *                   NODES,SHEAT,GAMMA,COLD,THIST,NVISC,SCREEN,
     *                   WEIGHT,TSTOP,STABF)
C
C        FREE FORMAT INPUT DRIVER
C        JOHN K. PRENTICE,  11 JULY 1986
C
C        THIS ROUTINE READS THE FREE FORMAT INPUT FILE TO DETERMINE
C        PARAMETERS FOR A CALCULATION.  BELOW IS GIVEN A LIST OF THE
C        INPUT PARAMETERS SHOWING THE USER INPUT WORD(S), THE FORTRAN
C        VARIABLE NAME LOADED, AND THE DESCRIPTION OF THE PARAMETER.
C
C        INPUT WORD(S)           FORTRAN    TYPE         DESCRIPTION
C                                VARIABLE
C                                NAME
C        -------------           --------   ----   ---------------------
C
C        PROBLEM_NUMBER            PROB      R     PROBLEM NUMBER
C            -OR-
C        PROB
C
C        TITLE                     TITLE     C     80 CHARACTER TITLE
C                                                  FOR THIS CALCULATION.
C                                                  WHEN THE WORD TITLE
C                                                  IS ENCOUNTERED IN
C                                                  INPUT, SCANNING WILL
C                                                  STOP FOR THAT LINE
C                                                  AND THE ENTIRE NEXT
C                                                  LINE WILL BE READ AS
C                                                  THE TITLE.
C
C        NUMBER_OF_NODES           NODES     I     NUMBER OF CELLS IN
C              -OR-                                MESH
C        NODES
C
C        CYCLE_STOP                CSTOP     I     LAST CYCLE TO RUN
C             -OR-                                 CALCUALATION TO
C        CSTOP
C
C        TIME_STOP                 TSTOP     R     LAST TIME TO RUN
C             -OR-                                 CALCULATION TO.
C        TSTOP                                     OPTIONAL.
C
C        STABILITY_FACTOR          STABF     R     COURANT CONDITION
C             -OR-                                 STABILITY FACTOR
C        STABF
C
C        FIRST_DUMP                FCYCLE    I     FIRST CYCLE TO
C             -OR-                                 DUMP RESULTS AT.
C        FCYCLE                                    DEFAULT IS 1
C
C        DUMP_INTERVAL             DCYCLE    I     CYCLE INTERVAL TO
C             -OR-                                 DUMP RESULTS.
C        DCYCLE                                    DEFAULT IS 1
C
C        SPECIFIC_HEAT             SHEAT     R     SPECIFIC HEAT OF GAS.
C              -OR-                                IF NOT SPECIFIED, THEN
C        SHEAT                                     A VALUE WILL BE CHOSEN
C                                                  APPROPIATE TO STEAM.
C                                                  IF SET, CONSTANT VALUE
C                                                  WILL BE ASSUMED. MUST
C                                                  SET GAMMA IS SPECIFIC
C                                                  HEAT IS SET.
C
C        GAMMA                     GAMMA     R     THERMODYNAMIC GAMMA OF
C                                                  GAS.  IF NOT SPECIFIED,
C                                                  A VALUE APPROPIATE TO
C                                                  STEAM WILL BE USED.
C                                                  IF SET, CONSTANT VALUE
C                                                  WILL BE ASSUMED.  MUST
C                                                  SET SPECIFIC HEAT IF
C                                                  GAMMA IS SET.
C
C        MOLECULAR_WEIGHT          WEIGHT    R     MOLEULAR WEIGHT IN
C             -OR-                                 KG/MOLE.  IF NOT
C        WEIGHT                                    SPECIFIED, A VALUE
C                                                  APPROPIATE TO WATER
C                                                  WILL BE USED
C
C        INCLUDE_SCREENS           SCREEN    B     IF SET, SCREENS WILL
C                                                  BE INCLUDED IN
C                                                  CALCULATION.  DEFAULT
C                                                  IS NO SCREENS
C
C        CONSTANT_FLOW             COLD      R     IF SET, CONSTANT FLOW
C                                                  THROUGH THE NOZZLE
C                                                  WILL BE ASSUMED AT THE
C                                                  MASS RATE GIVEN
C
C        CHAMBER_DENSITY           D0        R     INITIAL CHAMBER
C             -OR-                                 DENSITY (KG/M**3)
C        D0
C
C        CHAMBER_ENERGY            E0        R     INITIAL CHAMBER
C             -OR-                                 SPECIFIC INTERNAL
C        E0                                        ENERGY (J/KG)
C
C        TEMPERATURE_HISTOGRAM     THIST     B     CAUSE TEMPERATURE
C             -OR-                                 HISTOGRAM TO BE
C        THIST                                     PRODUCED.  DEFAULT
C                                                  IS NO PLOT
C
C        DENSITY_HISTOGRAM         DHIST     B     CAUSE
C             -OR-                                 DENSITY HISTOGRAM TO
C        DHIST                                     BE PRODUCED. DEFAULT
C                                                  IS NO PLOT
C
C        PRESSURE_HISTOGRAM        PHIST     B     CAUSE
C             -OR-                                 PRESSURE HISTOGRAM TO
C        PHIST                                     BE PRODUCED.  DEFAULT
C                                                  IS NO PLOT
C
C        VELOCITY_HISTOGRAM        VHIST     B     CAUSE
C             -OR-                                 VELOCITY HISTOGRAM TO
C        VHIST                                     BE PRODUCED.  DEFAULT
C                                                  IS NO PLOT
C
C        DEBUG                     DEBUG     B     CAUSE
C                                                  DEBUG INFORMATION TO
C                                                  BE PRODUCED EACH DUMP
C                                                  DEFAULT NO DEBUG INFO
C
C        STATIONS                  STATS     I     LIST OF CELL NUMBERS
C          -OR-                                    FOR WHICH TO STORE
C        STATS                                     STATION DATA.  NOTE
C                                                  THAT FORTRAN VARIABLE
C                                                  STATS IS AN INTEGER
C                                                  ARRAY.  THE LENGTH
C                                                  IS COMPUTED IN THIS
C                                                  ROUTINE TO BE NSTAT
C                                                  AND THIS VALUE IS
C                                                  PASSED BACK.  DEFAULT
C                                                  IS NSTAT=0.
C                                                  NOTE:  THE NUMBER OF
C                                                  WORDS OF
C                                                  CM NEEDED FOR EACH
C                                                  STATION IS NCORE
C
C        *** IF NSTAT > 0, THEN THE FOLLOWING TYPES OF STATION PLOTS
C            MAY BE REQUESTED.  DEFAULT IN ALL CASES IS NO PLOT
C
C        PRESSURE_PLOT             PPLOT     B     PRESSURE -VS- TIME
C             -OR-                                 PLOT FOR EACH
C        PPLOT                                     STATION
C
C        DENSITY_PLOT              DPLOT     B     DENSITY -VS- TIME
C             -OR-                                 PLOT FOR EACH
C        DPLOT                                     STATION
C
C        VELOCITY_PLOT             VPLOT     B     VELOCITY -VS- TIME
C             -OR-                                 PLOT FOR EACH
C        VPLOT                                     STATION
C
C        TEMPERATURE_PLOT          TPLOT     B     TEMPERATURE -VS- TIME
C             -OR-                                 PLOT FOR EACH STATION
C        TPLOT
C
C        PRINT_STATION_DATA        SLIST     B     PRINT STATION DATA
C             -OR-                                 FOR EACH STATION
C        SLIST
C
      IMPLICIT REAL (A-H,O-Z)
      INTEGER CSTOP,FCYCLE,DCYCLE,STATS(*),TYPE
      LOGICAL ABORT,DHIST,PHIST,VHIST,DEBUG,EOFF,PPLOT,DPLOT,VPLOT,
     *        THIST,TPLOT,SLIST,SCREEN
      CHARACTER*80 TITLE
      CHARACTER CARD*80,FIELD*80
      COMMON /CIMAGE/ CARD,FIELD
      COMMON /IMAGE/ EOFF,ICPNT
C                                      *********************************
C                                      DEFAULTS
C                                      *********************************
      NODES=-999
      COLD=-999.0
      SHEAT=-999.0
      GAMMA=-999.0
      TSTOP=1.E10
      STABF=0.5
      WEIGHT=18.016E-3
      DCYCLE=1
      FCYCLE=1
      NVISC=0
      SCREEN=.FALSE.
      THIST=.FALSE.
      DHIST=.FALSE.
      PHIST=.FALSE.
      VHIST=.FALSE.
      DEBUG=.FALSE.
      PPLOT=.FALSE.
      DPLOT=.FALSE.
      VPLOT=.FALSE.
      TPLOT=.FALSE.
      SLIST=.FALSE.
      NSTAT=0
C                                      *********************************
C                                      PARSE INPUT AND COMPARE TO
C                                      EXPECTED STRINGS
C                                      *********************************
   10 CALL NEXT
   20 IF(.NOT.EOFF) THEN
C                                      *********************************
C                                      PROBLEM NUMBER
C                                      *********************************
          IF (FIELD.EQ.'PROB'.OR.FIELD.EQ.'PROBLEM_NUMBER') THEN
              CALL VALUE (PROB,TYPE)
              IF(TYPE.NE.1) THEN
                  CALL TERROR (1,FIELD,TYPE)
                  STOP
              ENDIF
C                                      *********************************
C                                      TITLE
C                                      *********************************
          ELSEIF (FIELD.EQ.'TITLE') THEN
              READ(5,'(A)',END=30) TITLE
              ICPNT=9999
              GO TO 10
   30         WRITE(6,'('' END OF FILE ENCOUNTERED WHILE TRYING TO '',
     *                  ''READ THE TITLE, ABORT'')')
              STOP
C                                      *********************************
C                                      NUMBER OF NODES
C                                      *********************************
          ELSEIF (FIELD.EQ.'NUMBER_OF_NODES'.OR.FIELD.EQ.'NODES') THEN
              CALL VALUE(RNODES,TYPE)
              IF(TYPE.NE.1) THEN
                  CALL TERROR(1,FIELD,TYPE)
                  STOP
              ENDIF
              NODES=INT(RNODES)
C
C        TIME STOP
C
          ELSEIF (FIELD.EQ.'TIME_STOP'.OR.FIELD.EQ.'TSTOP') THEN
              CALL VALUE(TSTOP,TYPE)
              IF(TYPE.NE.1) THEN
                  CALL TERROR(1,FIELD,TYPE)
                  STOP
              ENDIF
C
C        STABILITY FACTOR
C
          ELSEIF (FIELD.EQ.'STABILITY_FACTOR'.OR.FIELD.EQ.'STABF') THEN
              CALL VALUE(STABF,TYPE)
              IF(TYPE.NE.1) THEN
                    CALL TERROR(1,FIELD,TYPE)
                    STOP
              ENDIF
C                                      *********************************
C                                      CYCLE STOP
C                                      *********************************
          ELSEIF (FIELD.EQ.'CSTOP'.OR.FIELD.EQ.'CYCLE_STOP') THEN
              CALL VALUE (RCSTOP,TYPE)
              IF(TYPE.NE.1) THEN
                  CALL TERROR (1,FIELD,TYPE)
                  STOP
              ENDIF
              CSTOP=INT(RCSTOP)
C                                      *********************************
C                                      FIRST DUMP
C                                      *********************************
          ELSEIF (FIELD.EQ.'FCYCLE'.OR.FIELD.EQ.'FIRST_DUMP') THEN
              CALL VALUE (RCYCLE,TYPE)
              IF(TYPE.NE.1) THEN
                  CALL TERROR (1,FIELD,TYPE)
                  STOP
              ENDIF
              FCYCLE=INT(RCYCLE)
C                                      *********************************
C                                      DUMP INTERVAL
C                                      *********************************
          ELSEIF (FIELD.EQ.'DCYCLE'.OR.FIELD.EQ.'DUMP_INTERVAL') THEN
              CALL VALUE (RCYCLE,TYPE)
              IF(TYPE.NE.1) THEN
                  CALL TERROR (1,FIELD,TYPE)
                  STOP
              ENDIF
              DCYCLE=INT(RCYCLE)
C                                      *********************************
C                                      SPECIFIC HEAT
C                                      *********************************
          ELSEIF (FIELD.EQ.'SPECIFIC_HEAT'.OR.FIELD.EQ.'SHEAT') THEN
              CALL VALUE (SHEAT,TYPE)
              IF(TYPE.NE.1) THEN
                  CALL TERROR (1,FIELD,TYPE)
                  STOP
              ENDIF
C                                      *********************************
C                                      MOLECULAR WEIGHT
C                                      *********************************
          ELSEIF (FIELD.EQ.'MOLECULAR_WEIGHT'.OR.FIELD.EQ.'WEIGHT') THEN
              CALL VALUE(WEIGHT,TYPE)
              IF(TYPE.NE.1) THEN
                  CALL TERROR (1,FIELD,TYPE)
                  STOP
              ENDIF
C                                      *********************************
C                                      GAMMA
C                                      *********************************
          ELSEIF (FIELD.EQ.'GAMMA') THEN
              CALL VALUE (GAMMA,TYPE)
              IF(TYPE.NE.1) THEN
                  CALL TERROR (1,FIELD,TYPE)
                  STOP
              ENDIF
C                                      *********************************
C                                      CONSTANT MASS FLOW
C                                      *********************************
          ELSEIF (FIELD.EQ.'CONSTANT_FLOW') THEN
              CALL VALUE (COLD,TYPE)
              IF(TYPE.NE.1) THEN
                  CALL TERROR (1,FIELD,TYPE)
                  STOP
              ENDIF
C                                      *********************************
C                                      MEAN FREE PATH IN SHOCK
C                                      *********************************
          ELSEIF (FIELD.EQ.'MEAN_FREE_PATH') THEN
              CALL VALUE(AVISC,TYPE)
              IF(TYPE.NE.1) THEN
                  CALL TERROR(1,FIELD,TYPE)
                  STOP
              ENDIF
              NVISC=INT(AVISC)
              IF(NVISC.LT.0) THEN
                  WRITE(6,'(//'' ILLEGAL MEAN FREE PATH OF '',I5,
     *                '', ABORT.''//)') NVISC
                  STOP
              ENDIF
C                                      *********************************
C                                      INITIAL CHAMBER DENSITY
C                                      *********************************
          ELSEIF (FIELD.EQ.'CHAMBER_DENSITY'.OR.FIELD.EQ.'D0') THEN
              CALL VALUE (D0,TYPE)
              IF(TYPE.NE.1) THEN
                  CALL TERROR (1,FIELD,TYPE)
                  STOP
              ENDIF
C                                      *********************************
C                                      INITIAL CHAMBER ENERGY
C                                      *********************************
          ELSEIF (FIELD.EQ.'CHAMBER_ENERGY'.OR.FIELD.EQ.'E0') THEN
              CALL VALUE (E0,TYPE)
              IF(TYPE.NE.1) THEN
                  CALL TERROR (1,FIELD,TYPE)
                  STOP
              ENDIF
C                                      *********************************
C                                      INCLUDE SCREENS
C                                      *********************************
          ELSEIF (FIELD.EQ.'INCLUDE_SCREENS') THEN
              SCREEN=.TRUE.
C                                      *********************************
C                                      TEMPERATURE HISTOGRAM
C                                      *********************************
          ELSEIF (FIELD.EQ.'THIST'.OR.FIELD.EQ.'TEMPERATURE_HISTOGRAM')
     *  THEN
              THIST=.TRUE.
C                                      *********************************
C                                      DENSITY HISTOGRAM
C                                      *********************************
          ELSEIF (FIELD.EQ.'DHIST'.OR.FIELD.EQ.'DENSITY_HISTOGRAM') THEN
              DHIST=.TRUE.
C                                      *********************************
C                                      PRESSURE HISTOGRAM
C                                      *********************************
          ELSEIF (FIELD.EQ.'PHIST'.OR.FIELD.EQ.'PRESSURE_HISTOGRAM')
     *    THEN
              PHIST=.TRUE.
C                                      *********************************
C                                      VELOCITY HISTOGRAM
C                                      *********************************
          ELSEIF (FIELD.EQ.'VHIST'.OR.FIELD.EQ.'VELOCITY_HISTOGRAM')
     *    THEN
              VHIST=.TRUE.
C                                      *********************************
C                                      DEBUG
C                                      *********************************
          ELSEIF (FIELD.EQ.'DEBUG') THEN
              DEBUG=.TRUE.
C                                      *********************************
C                                      STATIONS
C                                      *********************************
          ELSEIF (FIELD.EQ.'STATS'.OR.FIELD.EQ.'STATIONS') THEN
   40         NSTAT=NSTAT+1
              IF (NSTAT.GT.IMAX) THEN
                  WRITE(6,'('' TOO MANY STATIONS HAVE BEEN DEFINED.''/
     *                      '' THE MAXIMUM ALLOWED IS '',I5/
     *                      '' CHANGE IMAX IN THE'',
     *                      '' PROGRAM AND RECOMPILE'')') IMAX
                  STOP
              ELSE
                  CALL VALUE (RSTAT,TYPE)
                  IF (TYPE.NE.1) THEN
                      NSTAT=NSTAT-1
                      EOFF=.FALSE.
                      IF (TYPE.EQ.-1) EOFF=.TRUE.
                      GO TO 20
                  ELSE
                      STATS(NSTAT)=INT(RSTAT)
                      GO TO 40
                  ENDIF
              ENDIF
C                                      *********************************
C                                      STATION PRESSURE PLOT
C                                      *********************************
          ELSEIF (FIELD.EQ.'PPLOT'.OR.FIELD.EQ.'PRESSURE_PLOT') THEN
              PPLOT=.TRUE.
C                                      *********************************
C                                      STATION DENSITY PLOT
C                                      *********************************
          ELSEIF (FIELD.EQ.'DPLOT'.OR.FIELD.EQ.'DENSITY_PLOT') THEN
              DPLOT=.TRUE.
C                                      *********************************
C                                      STATION VELOCITY PLOT
C                                      *********************************
          ELSEIF (FIELD.EQ.'VPLOT'.OR.FIELD.EQ.'VELOCITY_PLOT') THEN
              VPLOT=.TRUE.
C                                      *********************************
C                                      STATION TEMPERATURE PLOT
C                                      *********************************
          ELSEIF (FIELD.EQ.'TPLOT'.OR.FIELD.EQ.'TEMPERATURE_PLOT') THEN
              TPLOT=.TRUE.
C                                      *********************************
C                                      PRINT STATION DATA
C                                      *********************************
          ELSEIF (FIELD.EQ.'SLIST'.OR.FIELD.EQ.'PRINT_STATION_DATA')THEN
              SLIST=.TRUE.
C                                      *********************************
C                                      UNRECOGNIZED WORD
C                                      *********************************
          ELSE
              WRITE(6,'('' UNRECOGNIZED WORD ENCOUNTERED IN INPUT''/
     *                  '' THE WORD WAS ----> '',A/'' ABORT''//)') FIELD
              STOP
          ENDIF
      ELSE
C                                      *********************************
C                                      MAKE SURE THE NUMBER OF NODES IS
C                                      A VALID NUMBER
C                                      *********************************
          IF(NODES.EQ.-999) THEN
              WRITE(6,'('' THE NUMBER OF NODES WAS NOT SPECIFIED IN '',
     *                  ''INPUT, ABORT'')')
              STOP
          ELSEIF (NODES.LE.0) THEN
              WRITE(6,'('' AN ILLEGAL VALUE WAS GIVEN FOR THE NUMBER '',
     *                  ''OF NODES, ABORT'')')
              STOP
          ELSEIF (NODES.GT.IMAX) THEN
              WRITE(6,'('' THE NUMBER OF NODES REQUESTED WAS '',I5/
     *                  '' THE MAXIMUM ALLOWED IS '',I5/
     *                  '' TO RUN MORE THAN THE MAXIMUM, CHANGE IMAX'',
     *                  '' IN THE CODE AND RECOMPILE. ABORT.''//)')
     *                   NODES,IMAX
              STOP
          ENDIF
C                                      *********************************
C                                      CHECK WHETHER BOTH GAMMA AND
C                                      SPECIFIC HEAT WERE GIVEN
C                                      *********************************
          IF(GAMMA.GT.0.0.OR.SHEAT.GT.0.0) THEN
              IF(GAMMA.GT.0.0.AND.SHEAT.LE.0.0) THEN
                  WRITE(6,'(//'' IF GAMMA IS SPECIFIED, SO MUST BE THE''
     *                       ,'' THE SPECIFIC HEAT.  ABORT'')')
                  STOP
              ELSEIF (GAMMA.LT.0.0.AND.SHEAT.GT.0.0) THEN
                  WRITE(6,'(//'' IF SPECIFIC HEAT IS SPECIFIED, SO '',
     *                  ''MUST BE GAMMA.  ABORT'')')
                  STOP
              ENDIF
          ENDIF
C                                      *********************************
C                                      SORT THE STATIONS AND BE SURE
C                                      THAT THERE IS ENOUGH STORAGE FOR
C                                      THE STATIONS
C                                      *********************************
          IF (NSTAT.GT.0) THEN
              NCYCLE=CSTOP-FCYCLE+1
              NCORE=NCYCLE*NSTAT
              IF (NCORE.GT.MAXSTA) THEN
              WRITE(6,'('' THERE IS NOT ENOUGH ARRAY STORAGE TO STORE ''
     *               ,''ALL THE STATION DATA FOR THE STATIONS DEFINED''
     *               /'' YOU NEED AT LEAST '',I8,'' WORDS, BUT HAVE'',
     *               '' ONLY '',I8/'' CHANGE PARAMETER MAXSTA IN THE ''
     *               ''MAIN ROUTINE TO AT LEAST '',I8,'' AND RECOMPILE''
     *               //)') NCORE,MAXSTA,NCORE
                  STOP
              ENDIF
              NCORE=NCYCLE
              ABORT=.FALSE.
              CALL QSORT (STATS,NSTAT)
              DO 99 I=1,NSTAT
                  IF(STATS(I).LT.1.OR.STATS(I).GT.NODES) WRITE(6,'(/
     *              '' STATION AT CELL '',I5,'' IS OUTSIDE MESH'')')
     *              STATS(I)
   99         CONTINUE
          ENDIF
C                                      *********************************
C                                      PRINT OUT INPUT VALUES
C                                      *********************************
      WRITE(6,'(''1CORTESA ONE DIMENSIONAL GAS DYNAMICS CODE''//1X,A//
     *          '' PARAMETERS FOR THIS RUN ARE:''/)') TITLE
      WRITE(6,'('' PROBLEM NUMBER = '',F10.5/
     *          '' NUMBER OF NODES = '',I5/
     *          '' CYCLE STOP = '',I5/
     *          '' TIME STOP = '',1PE12.5/
     *          '' COURANT CONDITION STABILITY FACTOR = '',0PF5.2/
     *          '' FIRST CYCLE TO DUMP AT = '',I5/
     *          '' DUMP INTERVAL = '',I5/)') PROB,NODES,CSTOP,TSTOP,
     *             STABF,FCYCLE,DCYCLE
      IF(COLD.LT.0.0) THEN
          WRITE(6,'(//'' CALCULATE NOZZLE FLOW USING PRESSURE-TIME HISTO
     *RY FROM INPUT''/)')
      ELSE
          WRITE(6,'(//'' CALCULATE NOZZLE FLOW AS CONSTANT AT '',
     *          1PE12.5,'' KG/SEC''/)') COLD
      ENDIF
      IF(SCREEN) THEN
          WRITE(6,'(/'' INCLUDE SCREENS IN CALCULATION''/)')
      ELSE
          WRITE(6,'(/'' NO SCREENS ARE PRESENT IN CALCULATION''/)')
      ENDIF
      IF(SHEAT.GT.0.0) THEN
          WRITE(6,'(//'' USE CONSTANT SPECIFIC HEAT OF '',1PE12.5,
     *           '' JOULES/DEG KELVIN/KILOGRAM'')') SHEAT
          WRITE(6,'('' USE CONSTANT GAMMA OF '',1PE12.5)') GAMMA
      ELSE
          WRITE(6,'(//'' USE THERMODYNAMICS APPROPIATE TO STEAM'')')
      ENDIF
      WRITE(6,'('' MOLECULAR WEIGHT = '',1PE12.5,'' KG/MOLE''/)') WEIGHT
          IF (DHIST) WRITE(6,'('' PLOT DENSITY HISTOGRAM ''/)')
          IF (PHIST) WRITE(6,'('' PLOT PRESSURE HISTOGRAM ''/)')
          IF (VHIST) WRITE(6,'('' PLOT VELOCITY HISTOGRAM ''/)')
          IF (DEBUG) WRITE(6,'('' PRINT DEBUG INFORMATION ''/)')
          IF (NSTAT.GT.0) THEN
              IF (PPLOT) WRITE(6,'('' PLOT STATION PRESSURE ''/)')
              IF (DPLOT) WRITE(6,'('' PLOT STATION DENSITY ''/)')
              IF (VPLOT) WRITE(6,'('' PLOT STATION VELOCITY ''/)')
              IF (TPLOT) WRITE(6,'('' PLOT STATION TEMPERATURE ''/)')
              IF (SLIST) WRITE(6,'('' PRINT STATION DATA ''/)')
              WRITE(6,'(///'' STATIONS ARE LOCATED AT CELLS:''/)')
          WRITE(6,'(''       '',I5/)') (STATS(I),I=1,NSTAT)
          ELSEIF (PPLOT.OR.DPLOT.OR.VPLOT.OR.TPLOT) THEN
          WRITE(6,'(//'' STATION PLOTS HAVE BEEN REQUESTED, BUT NO'',
     *                '' STATIONS HAVE BEEN DEFINED.  ABORT''//)')
              STOP
          ELSEIF (SLIST) THEN
          WRITE(6,'(//'' YOU HAVE REQUESTED A PRINTOUT OF THE STATION'',
     *                '' DATA, BUT NO STATIONS HAVE BEEN DEFINED. '',
     *                '' ABORT''//)')
              STOP
          ENDIF
          RETURN
      ENDIF
      GO TO 10
      END
