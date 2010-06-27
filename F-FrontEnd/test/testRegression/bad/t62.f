
      INTEGER FUNCTION  TEST( i )
C***********************************************************************
C                                                                      *
C              REPEAT AND TIME THE EXECUTION OF KERNEL i               *
C                                                                      *
C                    i  - Input integer;   Test Kernel Serial Number   *
C                 TEST  - Repetition Loop Counter, decremented to 0    *
C                                                                      *
C***********************************************************************
C
cANSI IMPLICIT  DOUBLE PRECISION (A-H,O-Z)
cIBM  IMPLICIT  REAL*8           (A-H,O-Z)
C
CLOX  REAL*8 SECOND
C
C/      PARAMETER( l1=   1001, l2=   101, l1d= 2*1001 )
C/      PARAMETER( l13= 64, l13h= 64/2, l213= 64+32, l813= 8*64 )
C/      PARAMETER( l14= 2048, l16= 75, l416= 4*75 , l21= 25)
C
C/      PARAMETER( kn= 47, kn2= 95, np= 3, ls= 3*47, krs= 24)
C
      COMMON /SPACES/ ion,j5,k2,k3,MULTI,laps,Loop,m,kr,LP,n13h,ibuf,nx,
     1 L,npass,nfail,n,n1,n2,n13,n213,n813,n14,n16,n416,n21,nt1,nt2,
     2 last,idebug,mpy,Loops2,mucho,mpylim, intbuf(16)
C
      INTEGER    E,F,ZONE
      COMMON /ISPACE/ E(96), F(96),
     1  IX(1001), IR(1001), ZONE(300)
C
      COMMON /SPACER/ A11,A12,A13,A21,A22,A23,A31,A32,A33,
     2                AR,BR,C0,CR,DI,DK,
     3  DM22,DM23,DM24,DM25,DM26,DM27,DM28,DN,E3,E6,EXPMAX,FLX,
     4  Q,QA,R,RI,S,SCALE,SIG,STB5,T,XNC,XNEI,XNM
C
      DIMENSION     ZX(1023), XZ(1500)
      EQUIVALENCE ( ZX(1), Z(1)), ( XZ(1), X(1))
C
      COMMON /SPACE1/ U(1001), V(1001), W(1001),
     1  X(1001), Y(1001), Z(1001), G(1001),
     2  DU1(101), DU2(101), DU3(101), GRD(1001), DEX(1001),
     3  XI(1001), EX(1001), EX1(1001), DEX1(1001),
     4  VX(1001), XX(1001), RX(1001), RH(2048),
     5  VSP(101), VSTP(101), VXNE(101), VXND(101),
     6  VE3(101), VLR(101), VLIN(101), B5(101),
     7  PLAN(300), D(300), SA(101), SB(101)
C
      COMMON /SPACE2/ P(4,512), PX(25,101), CX(25,101),
     1  VY(101,25), VH(101,7), VF(101,7), VG(101,7), VS(101,7),
     2  ZA(101,7)  , ZP(101,7), ZQ(101,7), ZR(101,7), ZM(101,7),
     3  ZB(101,7)  , ZU(101,7), ZV(101,7), ZZ(101,7),
     4  B(64,64), C(64,64), H(64,64),
     5  U1(5,101,2),  U2(5,101,2),  U3(5,101,2)
C
C
      COMMON /BASER/ A110,A120,A130,A210,A220,A230,A310,A320,A330,
     2                AR0,BR0,C00,CR0,DI0,DK0,
     3  DM220,DM230,DM240,DM250,DM260,DM270,DM280,DN0,E30,E60,EXPMAX0,
     4  FLX0,Q0,QA0,R0,RI0,S0,SCALE0,SIG0,STB50,T0,XNC0,XNEI0,XNM0
C
      COMMON /BASE1/ U0(1001), V0(1001), W0(1001),
     1  X0(1001), Y0(1001), Z0(1001), G0(1001),
     2  DU10(101), DU20(101), DU30(101), GRD0(1001), DEX0(1001),
     3  XI0(1001), EX0(1001), EX10(1001), DEX10(1001),
     4  VX0(1001), XX0(1001), RX0(1001), RH0(2048),
     5  VSP0(101), VSTP0(101), VXNE0(101), VXND0(101),
     6  VE30(101), VLR0(101), VLIN0(101), B50(101),
     7  PLAN0(300), D0(300), SA0(101), SB0(101)
C
      COMMON /BASE2/ P0(4,512), PX0(25,101), CX0(25,101),
     1  VY0(101,25), VH0(101,7), VF0(101,7), VG0(101,7), VS0(101,7),
     2  ZA0(101,7)  , ZP0(101,7), ZQ0(101,7), ZR0(101,7), ZM0(101,7),
     3  ZB0(101,7)  , ZU0(101,7), ZV0(101,7), ZZ0(101,7),
     4  B0(64,64), CC0(64,64), H0(64,64),
     5  U10(5,101,2),  U20(5,101,2),  U30(5,101,2)
C
      COMMON /TAU/   tclock, tsecov, testov, cumtim(4)
C
      COMMON /SPACE0/ TIME(47), CSUM(47), WW(47), WT(47), ticks,
     1                FR(9), TERR1(47), SUMW(7), START,
     2              SKALE(47), BIAS(47), WS(95), TOTAL(47), FLOPN(47),
     3                IQ(7), NPF, NPFS1(47)
C
C
C*******************************************************************************
C         Repeat execution of each Kernel(i) :     DO 1 L= 1,Loop   etc.
C*******************************************************************************
C
C    From the beginning in 1970 each sample kernel was executed just
C    once since supercomputers had high resolution, microsecond clocks.
C    In 1982 a repetition Loop was placed around each of the 24 LFK
C    kernels in order to run each kernel long enough for accurate
C    timing on mini-computer systems with poor cpu-clock resolution since
C    the majority of systems could only measure cpu-time to 0.01 seconds.
C    By 1990 however, several compilers' optimizers were factoring or
C    hoisting invariant computation outside some repetition Loops thus
C    distorting those Fortran samples.  The effect was usually absurd
C    Mflop rates which had to be corrected with compiler directives.
C    Therefore, in April 1990 these repetition Loops were removed from
C    subroutine KERNEL and submerged in subroutine TEST beyond the scope
C    of compiler optimizations.   Thus the 24 samples are now foolproof
C    and it will no longer be necessary to double check the machine code.
C
C    Very accurate, convergent methods have been developed to measure the
C    overhead time used for subroutines SECOND and TEST in subroutines
C    SECOVT and TICK respectively.  Thus, the LFK test may use substantially
C    more cpu time on systems with poor cpu-clock resolution.
C    The 24 C verison tests in CERNEL have also been revised to correspond with
C    the Fortran KERNEL. The 24 computation samples have NOT been changed.
C
C*******************************************************************************
C
cbug  IF( (LP.NE.Loop).OR.(L.LT.1).OR.(L.GT.Loop)) THEN
cbug      CALL TRACE('TEST    ')
cbug      CALL WHERE(0)
cbug  ENDIF
C                                    Repeat kernel test:   Loop times.
      IF( L .LT. Loop )  THEN
          L    = L + 1
          TEST = L
          RETURN
      ENDIF
C                                    Repeat kernel test:   Loop*Loops2
          ik   = i
      IF( mpy .LT. Loops2 )  THEN
          mpy  = mpy + 1
          nn   = n
C
           IF( i.EQ.0 ) GO TO 100
           IF( i.LT.0 .OR. i.GT.24 )  THEN
               CALL TRACE('TEST    ')
               CALL WHERE(0)
           ENDIF
C                   RE-INITIALIZE OVER-STORED INPUTS:
C
        GO TO( 100,   2, 100,   4,   5,   6, 100, 100,
     .         100,  10, 100, 100,  13,  14, 100,  16,
     .          17,  18,  19,  20,  21, 100,  23, 100, 100  ),  i
C
C     When MULTI.GE.100 each kernel is executed over a million times
C     and the time used to re-intialize overstored input variables
C     is negligible.  Thus each kernel may be run arbitrarily many times
C     (MULTI >> 100) without overflow and produce verifiable checksums.
C
C***********************************************************************
C
    2 DO 200 k= 1,nn
  200 X(k)= X0(k)
      GO TO 100
C***************************************
C
    4        m= (1001-7)/2
      DO 400 k= 7,1001,m
  400 XZ(k)= X0(k)
      GO TO 100
C***************************************
C
    5 DO 500 k= 1,nn
  500 X(k)= X0(k)
      GO TO 100
C***************************************
C
    6 DO 600 k= 1,nn
  600 W(k)= W0(k)
      GO TO 100
C***************************************
C
   10 DO 1000 k= 1,nn
      DO 1000 j= 5,13
 1000   PX(j,k)= PX0(j,k)
      GO TO 100
C***************************************
C
   13 DO 1300 k= 1,nn
         P(1,k)= P0(1,k)
         P(2,k)= P0(2,k)
         P(3,k)= P0(3,k)
 1300    P(4,k)= P0(4,k)
c
      DO 1301 k= 1,64
      DO 1301 j= 1,64
 1301    H(j,k)= H0(j,k)
      GO TO 100
C***************************************
C
   14 DO 1400   k= 1,nn
      RH(IR(k)  )= RH0(IR(k)  )
 1400 RH(IR(k)+1)= RH0(IR(k)+1)
      GO TO 100
C***************************************
C
   16 k2= 0
      k3= 0
      GO TO 100
C***************************************
C
   17 DO 1700 k= 1,nn
 1700     VXNE(k)= VXNE0(k)
      GO TO 100
C***************************************
C
   18 DO 1800 k= 2,6
      DO 1800 j= 2,nn
        ZU(j,k)= ZU0(j,k)
        ZV(j,k)= ZV0(j,k)
        ZR(j,k)= ZR0(j,k)
 1800   ZZ(j,k)= ZZ0(j,k)
      GO TO 100
C***************************************
C
   19 STB5= STB50
      GO TO 100
C***************************************
C
   20 XX(1)= XX0(1)
      GO TO 100
C***************************************
C
   21 DO 2100 k= 1,nn
      DO 2100 j= 1,25
 2100   PX(j,k)= PX0(j,k)
      GO TO 100
C***************************************
C
   23 DO 2300 k= 2,6
      DO 2300 j= 2,nn
 2300   ZA(j,k)= ZA0(j,k)
C***********************************************************************
C
  100 CONTINUE
C
          L    = 1
          TEST = 1
          RETURN
      ENDIF
C
          mpy  = 1
          L    = 1
          TEST = 0
C                                   switchback to TICK to measure testov
           IF( i.EQ. (-73))  RETURN
C
C***********************************************************************
C           t= second(0)  := cumulative cpu time for task in seconds.
C***********************************************************************
C
      cumtim(1)= 0.0d0
C         TEMPUS= SECOND( cumtim(1)) - START
C
      CALL TRACE ('TEST    ')
CPFM      ikern= i
CPFM      call ENDPFM(ion)
C$C                           5 get number of page faults (optional)
C$      KSTAT= LIB$STAT_TIMER(5,KPF)
C$      NPF  = KPF - IPF
C
C
C                             Checksum results; re-initialize all inputs
      CALL TESTS ( i, TEMPUS )
C
C
C$C                           5 get number of page faults (optional) VAX
C$      NSTAT= LIB$STAT_TIMER(5,IPF)
C
CPFM       IF( INIPFM( ion, 0) .NE. 0 )  THEN
CPFM           CALL WHERE(20)
CPFM       ENDIF
      CALL TRACK ('TEST    ')
C
C      The following pause can be used for stop-watch timing of each kernel.
C      You may have to increase the iteration count MULTI in Subr. VERIFY.
C
C/           PAUSE
C
      mpy   = 1
      mpylim= Loops2
      L     = 1
      LP    = Loop
      ik    = i+1
      TEST  = 0
      cumtim(1)= 0.0d0
C      START= SECOND( cumtim(1))
      RETURN
C
C$      DATA  IPF/0/, KPF/0/
      END
