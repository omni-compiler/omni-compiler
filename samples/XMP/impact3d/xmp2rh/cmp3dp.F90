!----------------------------------------------------------------------
! compute pressure
!
      subroutine cmp3dp
!
!   <arguments>
!     none
!
!   <remarks>
!     none
!
!    coded by sakagami,h. ( isr ) 88/06/29
! modified by Sakagami,H. ( NIFS ) 13/10/18 : xmp benchmark version
!----------------------------------------------------------------------
#include "phys.macro"
      use parameter
      use constant
      use phys
      include "implicit.h"
      save
!....
      call measure( 2, ' ', 5 )
!....
!$XMP LOOP (iy,iz) ON t(*,iy,iz)
!$OMP PARALLEL DO SCHEDULE(STATIC) PRIVATE(iy,ix)
      do iz = 1, lz
      do iy = 1, ly
      do ix = 1, lx
         sp(ix,iy,iz) = ( sgam - 1.0d0 ) * ( se(ix,iy,iz) - &
           0.5d0 * ( sm(ix,iy,iz)**2 + &
                   sn(ix,iy,iz)**2 + sl(ix,iy,iz)**2 ) / sr(ix,iy,iz) )
      end do
      end do
      end do
!....
      call measure( 3, ' ', 5 )
!....
      return
      end
