!----------------------------------------------------------------------
! initialize physical values
!
      subroutine init 
!
!   <arguments>
!     none
!
!   <remarks>
!     none
!
!     coded by sakagami,h. ( himeji-tech.ac.jp ) 95/12/13
! modified by Sakagami,H. ( NIFS ) 13/10/18 : xmp benchmark version
!----------------------------------------------------------------------
#include "phys.macro"
      use parameter
      use constant
      use phys
      include "implicit.h"
      save
!....
!$OMP PARALLEL DO SCHEDULE(STATIC) PRIVATE(iy,ix)
      do iz = 1, llz
      do iy = 1, ly
      do ix = 1, lx
         sr(ix,iy,iz) = 100.0d0
         sp(ix,iy,iz) = 1000.0d0
         sm(ix,iy,iz) = 0.0d0
         sn(ix,iy,iz) = 0.0d0
         sl(ix,iy,iz) = 0.0d0
      end do
      end do
      end do
!..
      izs = max( lz/4 + 1, llbz )
      izs = izs - llbz + 1
      ize = min( lz/4 * 3, lubz )
      ize = ize - llbz + 1
!.
!$OMP PARALLEL DO SCHEDULE(STATIC) PRIVATE(iy,ix) FIRSTPRIVATE(izs,ize)
      do iz = izs, ize
      do iy = ly/4 + 1, ly/4 * 3
      do ix = lx/4 + 1, lx/4 * 3
	 sr(ix,iy,iz) = 2.0d0
	 sp(ix,iy,iz) = 5.0d0
      end do
      end do
      end do
!....
!$OMP PARALLEL DO SCHEDULE(STATIC) PRIVATE(iy,ix)
      do iz = 1, llz
      do iy = 1, ly
      do ix = 1, lx
         se(ix,iy,iz) = sp(ix,iy,iz) / ( sgam - 1.0d0 ) + &
            0.5d0 * ( sm(ix,iy,iz)**2 + sn(ix,iy,iz)**2 +  &
                    sl(ix,iy,iz)**2 ) / sr(ix,iy,iz)
      end do
      end do
      end do
!....
      return
      end
