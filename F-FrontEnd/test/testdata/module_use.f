
       Module razmer
        Parameter (n1x=50, n1y=20, N11=n1x*n1y, M11=2*N11)
        Parameter (n0=Max(n1x,n1y), nn=n0*n0, mm=n0+4)
      End Module razmer 

      module DynamParamM
        Type DynamParamT
           real(8) :: GrStrt, GrEnd, GrMul, Eps
           integer :: Nsig, ItMax
        End Type DynamParamT
        type(DynamParamT) DynamParam
      end module DynamParamM

        Module Parameters
          Real*8  W, Gr, Pr
          Integer Itype
        End Module Parameters

        Module Numbers
          Integer  N, K, N2, K2, KN, NS2
        End Module Numbers

        Module Sizes
          Integer  NX, NY, NKX, NKY
        End Module Sizes

        Module BifPar
          Integer  Mth, Jjob
          Real*8   U, Omega
        End Module BifPar


        Module  FNS
          Use razmer

          Real*8, Dimension(0:mm) ::                                    &
     *              F1, F2, F3, F4, G1, G2, G3, G4, A, B
        End Module FNS

        Module FHT 
          Use razmer

           Real*8, Dimension(0:mm) :: V1, V2, T1, T2
        End Module FHT


        Module ProdL
           Use razmer

            Real*8, Dimension(-1:mm,-1:mm) ::  FF, F2F, WW, W2W, FBW
        End Module ProdL

        Module ProdN
           Use razmer

          Real*8  F1FF( 0:mm, 0:mm, 0:mm),  FWF( 0:mm,-1:mm, 0:mm)
          Real*8   WWW(-1:mm,-1:mm,-1:mm), W1FW(-1:mm, 0:mm,-1:mm)
        End Module ProdN


        Module NsLin
           Use razmer

            Real*8, Dimension(mm,mm) ::                                 &
     *            VXX,  VXY,  VYX,  VYY,  VXX2, VXY2, VYX2, VYY2
        End Module NsLin

        Module NsArh
           Use razmer

            Real*8, Dimension(mm,mm) ::  WTX,  WTY,  QXT,  QYT
        End Module NsArh

        Module NsNel
           Use razmer

            Real*8, Dimension(mm,mm,mm) ::                              &
     *            VXXX, VXXY, VXYX, VXYY, VYXX, VYXY, VYYX, VYYY
        End Module NsNel

        Module NsMf
           Use razmer

            Real*8, Dimension(mm,mm) ::  VXXMF, VYYMF
        End Module NsMf

        Module HtLin
           Use razmer
 
            Real*8, Dimension(mm,mm) :: TXX, TYY, TXX2, TYY2,           &
     *                                 PXX, PYY, PXX2, PYY2 
        End Module HtLin

        Module HtNel
           Use razmer

            Real*8, Dimension(mm,mm,mm) ::                              &
     *             WXTX, WXTY, WYTX, WYTY, WXPX, WXPY, WYPX, WYPY
        End Module HtNel

        Module ENS
           Use razmer

           Real*8, Dimension(N11,N11) :: EnsLap, EnsArh
           Real*8, Dimension(N11)     :: EnsFbt
        End Module ENS

        Module Fmag
           Use razmer

           Real*8, Dimension(N11,N11) :: EnsMx, EnsMy, EnsMxy
        End Module Fmag

        Module EHT
           Use razmer

           Real*8, Dimension(N11,N11) :: HtLap, HtVbt
           Real*8, Dimension(N11)     :: HtLbt
           Real*8, Dimension(n0,n0)   :: BT
        End Module EHT

        Module Ttime
           Use razmer

           Real*8, Dimension(N11,N11) ::  EnsTim, HtTim
        End Module Ttime

        Module DumBilin
          Use razmer

          Real*8,  Dimension(N11,N11) ::                                &
     *           POP, Poj1, Poj2, Poj3, Poj4

          Real*8, Dimension(nn) :: POP1
        End Module DumBilin
