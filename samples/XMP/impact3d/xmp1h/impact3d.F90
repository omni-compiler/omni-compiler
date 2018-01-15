!----------------------------------------------------------------------
! main program for 3-d tvd scheme
!
!   <remarks>
!     none
!
!    coded by sakagami,h. ( isr ) 90/09/08
! modified by Sakagami,H. ( NIFS ) 12/12/12 : test version
! modified by Sakagami,H. ( NIFS ) 13/10/18 : xmp benchmark version
!----------------------------------------------------------------------
#include "phys.macro"
      use parameter
      use constant
      use phys
      include "implicit.h"
!$    integer OMP_GET_MAX_THREADS
!....
      integer xmp_node_num, xmp_all_num_nodes
!....
      if( xmp_all_num_nodes() .ne. lnpz ) then
          write(*,*) 'ERROR: XANN must match with lnpz.', &
                                  xmp_all_num_nodes(), lnpz
          go to 999
      end if 
!....
      call measure ( 1, 'total', 1 )
      call measure ( 1, 'adv3dx', 2 )
      call measure ( 1, 'adv3dy', 3 )
      call measure ( 1, 'adv3dz', 4 )
      call measure ( 1, 'cmp3dp', 5 )
      call measure ( 1, 'cmpram', 6 )
      call measure ( 1, 'reflect in z', 7 )
!....
      sgam = 5.0d0 / 3.0d0
      sram = 0.0d0
      smue = 0.95d0
      seps = 0.2d0
      somga = 2.0d0
      szero = 1.0d-5
      szero2 = 1.0d-10
      wdltt = 0.1d0
      wtime = 0.0d0
!....                                                       * initial *
      call init
!....
      call measure( 2, ' ', 1 )
!..
      do iloop = 1, lstep
!..
      call cmpram
      wtime = wtime + sram * wdltt
!..                                                     * advance x/4 *
      call adv3dx ( 0.25d0 )
      call cmp3dp
!..                                                     * advance y/2 *
      call adv3dy ( 0.5d0 )
      call cmp3dp 
!..                                                     * advance z/2 *
      call adv3dz ( 0.5d0 )
      call cmp3dp 
!..                                                     * advance x/2 *
      call adv3dx ( 0.5d0 )
      call cmp3dp 
!..                                                     * advance y/2 *
      call adv3dy ( 0.5d0 )
      call cmp3dp 
!..                                                     * advance z/2 *
      call adv3dz ( 0.5d0 )
      call cmp3dp 
!..                                                     * advance x/4 *
      call adv3dx ( 0.25d0 )
      call cmp3dp 
!....
      end do
!....
      call measure( 3, ' ', 1 )
!....
      wcheck = 0.0d0
! !$XMP LOOP (iz) ON t(*,*,iz) REDUCTION(+:wcheck)
!$XMP LOOP (iz) ON t(*,*,iz)
      do iz = lz/4 + 1, lz/4 * 3
      do iy = ly/4 + 1, ly/4 * 3
      do ix = lx/4 + 1, lx/4 * 3
         wcheck = wcheck + sr(ix,iy,iz)
      end do
      end do
      end do
!$XMP REDUCTION(+:wcheck)
!..
! !$XMP TASK ON proc(1)
      if( xmp_node_num() .eq. 1 ) then
        write(*,*) '--------------------------------------------------'
        write(*,*) 'xmp1h, lnpz = ', lnpz
!$      write(*,*) 'threads    = ', OMP_GET_MAX_THREADS()
        write(*,*) 'lx, ly, lz = ', lx, ly, lz
        write(*,*) 'lstep      = ', lstep
        write(*,*) 'wtime, wcheck = ', wtime, wcheck
        call measure ( 0, ' ', 0 )
      end if
! !$XMP END TASK
!....
  999 continue
!     stop
      end
