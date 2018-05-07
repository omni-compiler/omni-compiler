module phys
      use parameter
      real*8 :: physval1(6,lx,ly,llz+1)
!     real*8 :: sr(lx,ly,llz+1), sm(lx,ly,llz+1), &
!               sp(lx,ly,llz+1), se(lx,ly,llz+1), &
!               sn(lx,ly,llz+1), sl(lx,ly,llz+1)
      real*8 :: physval2(5,lx,ly,llz)
!     real*8 :: wnue1(lx,ly,llz), wnue2(lx,ly,llz), &
!               wnue3(lx,ly,llz), wnue4(lx,ly,llz), &
!               wnue5(lx,ly,llz)
      real*8 :: physval3(5,lx,ly,0:llz)
!     real*8 :: walfa1(lx,ly,0:llz), walfa2(lx,ly,0:llz), &
!               walfa3(lx,ly,0:llz), walfa4(lx,ly,0:llz), &
!               walfa5(lx,ly,0:llz)
      real*8 :: physval4(8,lx,ly,llz+1)
!     real*8 :: wg1(lx,ly,llz+1), wg2(lx,ly,llz+1), &
!               wg3(lx,ly,llz+1), wg4(lx,ly,llz+1), &
!               wg5(lx,ly,llz+1)
!     real*8 :: wtmp1(lx,ly,llz+1), wtmp2(lx,ly,llz+1), &
!               wtmp3(lx,ly,llz+1)
      real*8 :: physval5(5,lx,ly,0:llz)
!     real*8 :: wff1(lx,ly,0:llz), wff2(lx,ly,0:llz), &
!               wff3(lx,ly,0:llz), wff4(lx,ly,0:llz), &
!               wff5(lx,ly,0:llz)
end module phys
