module phys
      use parameter
      real*8 :: physval1(6,lx,lly+1,llz+1)
!     real*8 :: sr(lx,lly+1,llz+1), sm(lx,lly+1,llz+1), &
!               sp(lx,lly+1,llz+1), se(lx,lly+1,llz+1), &
!               sn(lx,lly+1,llz+1), sl(lx,lly+1,llz+1)
      real*8 :: physval2(5,lx,lly,llz)
!     real*8 :: wnue1(lx,lly,llz), wnue2(lx,lly,llz), &
!               wnue3(lx,lly,llz), wnue4(lx,lly,llz), &
!               wnue5(lx,lly,llz)
      real*8 :: physval3(5,lx,0:lly,0:llz)
!     real*8 :: walfa1(lx,0:lly,0:llz), walfa2(lx,0:lly,0:llz), &
!               walfa3(lx,0:lly,0:llz), walfa4(lx,0:lly,0:llz), &
!               walfa5(lx,0:lly,0:llz)
      real*8 :: physval4(8,lx,lly+1,llz+1)
!     real*8 :: wg1(lx,lly+1,llz+1), wg2(lx,lly+1,llz+1), &
!               wg3(lx,lly+1,llz+1), wg4(lx,lly+1,llz+1), &
!               wg5(lx,lly+1,llz+1)
!     real*8 :: wtmp1(lx,lly+1,llz+1), wtmp2(lx,lly+1,llz+1), &
!               wtmp3(lx,lly+1,llz+1)
      real*8 :: physval5(5,lx,0:lly,0:llz)
!     real*8 :: wff1(lx,0:lly,0:llz), wff2(lx,0:lly,0:llz), &
!               wff3(lx,0:lly,0:llz), wff4(lx,0:lly,0:llz), &
!               wff5(lx,0:lly,0:llz)
end module phys
