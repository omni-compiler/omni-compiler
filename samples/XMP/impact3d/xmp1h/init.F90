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
!$XMP LOOP (iz) ON t(*,*,iz)
!$OMP PARALLEL DO SCHEDULE(STATIC) PRIVATE(iy,ix)
      do iz = 1, lz
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
!$XMP LOOP (iz) ON t(*,*,iz)
!$OMP PARALLEL DO SCHEDULE(STATIC) PRIVATE(iy,ix)
      do iz = lz/4 + 1, lz/4 * 3
      do iy = ly/4 + 1, ly/4 * 3
      do ix = lx/4 + 1, lx/4 * 3
	 sr(ix,iy,iz) = 2.0d0
	 sp(ix,iy,iz) = 5.0d0
      end do
      end do
      end do
!....
!$XMP LOOP (iz) ON t(*,*,iz)
!$OMP PARALLEL DO SCHEDULE(STATIC) PRIVATE(iy,ix)
      do iz = 1, lz
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
