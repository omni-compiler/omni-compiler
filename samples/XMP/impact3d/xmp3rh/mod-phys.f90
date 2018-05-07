module phys
      use parameter
!$XMP NODES proc(lnpx,lnpy,lnpz) 
!$XMP TEMPLATE t(lx,ly,lz)
!$XMP DISTRIBUTE t(BLOCK,BLOCK,BLOCK) ONTO proc
!....
      real*8 :: physval1(6,lx,ly,lz)
!     real*8 :: sr(lx,ly,lz), sm(lx,ly,lz), sp(lx,ly,lz), &
!               se(lx,ly,lz), sn(lx,ly,lz), sl(lx,ly,lz)
      real*8 :: physval2(5,lx,ly,lz)
!     real*8 :: wnue1(lx,ly,lz), wnue2(lx,ly,lz), wnue3(lx,ly,lz), &
!               wnue4(lx,ly,lz), wnue5(lx,ly,lz)
      real*8 :: physval3(5,lx,ly,lz)
!     real*8 :: walfa1(lx,ly,lz), walfa2(lx,ly,lz), &
!               walfa3(lx,ly,lz), walfa4(lx,ly,lz), &
!               walfa5(lx,ly,lz)
      real*8 :: physval4(8,lx,ly,lz)
!     real*8 :: wg1(lx,ly,lz), wg2(lx,ly,lz), wg3(lx,ly,lz), &
!               wg4(lx,ly,lz), wg5(lx,ly,lz)
!     real*8 :: wtmp1(lx,ly,lz), wtmp2(lx,ly,lz), wtmp3(lx,ly,lz)
      real*8 :: physval5(5,lx,ly,lz)
!     real*8 :: wff1(lx,ly,lz), wff2(lx,ly,lz), wff3(lx,ly,lz), &
!               wff4(lx,ly,lz), wff5(lx,ly,lz)
!....
!$XMP ALIGN (*,i,j,k) WITH t(i,j,k) :: &
!$XMP           physval1, physval2, physval3, physval4, physval5
!$XMP SHADOW (0,0:1,0:1,0:1) :: &
!$XMP           physval1, physval4
!$XMP SHADOW (0,1:0,1:0,1:0) :: &
!$XMP           physval3, physval5
end module phys
