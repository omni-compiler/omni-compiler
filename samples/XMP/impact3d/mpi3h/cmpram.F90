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
!$OMP PARALLEL DO SCHEDULE(STATIC) REDUCTION(max:wram) &
!$OMP    PRIVATE(iy,ix,wuu,wvv,www,wcc)
      do iz = 1, llz
      do iy = 1, lly
      do ix = 1, llx
         wuu = sm(ix,iy,iz) / sr(ix,iy,iz)
         wvv = sn(ix,iy,iz) / sr(ix,iy,iz)
         www = sl(ix,iy,iz) / sr(ix,iy,iz)
         wcc = sqrt( sgam * sp(ix,iy,iz) / sr(ix,iy,iz) )
         wram = max( wram, abs(wuu)+wcc, abs(wvv)+wcc, abs(www)+wcc )
      end do
      end do
      end do
!..
       call MPI_ALLREDUCE( wram, sram, 1, MPI_DOUBLE_PRECISION, &
                           MPI_MAX, MPI_COMM_WORLD, ierr )
!....
      sram = smue / sram
!....
      call measure( 3, ' ', 6 )
!....
      return
      end
