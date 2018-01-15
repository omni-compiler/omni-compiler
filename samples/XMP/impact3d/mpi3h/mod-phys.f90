module phys
      use parameter
      real*8 :: physval1(6,llx+1,lly+1,llz+1)
!     real*8 :: sr(llx+1,lly+1,llz+1), sm(llx+1,lly+1,llz+1), &
!               sp(llx+1,lly+1,llz+1), se(llx+1,lly+1,llz+1), &
!               sn(llx+1,lly+1,llz+1), sl(llx+1,lly+1,llz+1)
      real*8 :: physval2(5,llx,lly,llz)
!     real*8 :: wnue1(llx,lly,llz), wnue2(llx,lly,llz), &
!               wnue3(llx,lly,llz), wnue4(llx,lly,llz), &
!               wnue5(llx,lly,llz)
      real*8 :: physval3(5,0:llx,0:lly,0:llz)
!     real*8 :: walfa1(0:llx,0:lly,0:llz), walfa2(0:llx,0:lly,0:llz), &
!               walfa3(0:llx,0:lly,0:llz), walfa4(0:llx,0:lly,0:llz), &
!               walfa5(0:llx,0:lly,0:llz)
      real*8 :: physval4(8,llx+1,lly+1,llz+1)
!     real*8 :: wg1(llx+1,lly+1,llz+1), wg2(llx+1,lly+1,llz+1), &
!               wg3(llx+1,lly+1,llz+1), wg4(llx+1,lly+1,llz+1), &
!               wg5(llx+1,lly+1,llz+1)
!     real*8 :: wtmp1(llx+1,lly+1,llz+1), wtmp2(llx+1,lly+1,llz+1), &
!               wtmp3(llx+1,lly+1,llz+1)
      real*8 :: physval5(5,0:llx,0:lly,0:llz)
!     real*8 :: wff1(0:llx,0:lly,0:llz), wff2(0:llx,0:lly,0:llz), &
!               wff3(0:llx,0:lly,0:llz), wff4(0:llx,0:lly,0:llz), &
!               wff5(0:llx,0:lly,0:llz)
end module phys
