!----------------------------------------------------------------------
! compute rambda ( time increment )
!
      subroutine cmpram
!
!   <arguments>
!     none
!
!   <remarks>
!     none
!
!    coded by sakagami,h. ( isr ) 88/06/30
! modified by Sakagami,H. ( NIFS ) 13/10/18 : xmp benchmark version
!----------------------------------------------------------------------
#include "phys.macro"
      use parameter
      use constant
      use phys
      include "implicit.h"
      save
!....
      call measure( 2, ' ', 6 )
!....
      wram = 0.0d0
!....
!$XMP LOOP (ix,iy,iz) ON t(ix,iy,iz) REDUCTION(max:wram)
!$OMP PARALLEL DO SCHEDULE(STATIC) REDUCTION(max:wram) &
!$OMP    PRIVATE(iy,ix,wuu,wvv,www,wcc)
      do iz = 1, lz
      do iy = 1, ly
      do ix = 1, lx
         wuu = sm(ix,iy,iz) / sr(ix,iy,iz)
         wvv = sn(ix,iy,iz) / sr(ix,iy,iz)
         www = sl(ix,iy,iz) / sr(ix,iy,iz)
         wcc = sqrt( sgam * sp(ix,iy,iz) / sr(ix,iy,iz) )
         wram = max( wram, abs(wuu)+wcc, abs(wvv)+wcc, abs(www)+wcc )
      end do
      end do
      end do
!....
      sram = smue / wram
!....
      call measure( 3, ' ', 6 )
!....
      return
      end
