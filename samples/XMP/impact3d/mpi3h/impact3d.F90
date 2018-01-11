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
      call MPI_INIT( ierr )
      call MPI_COMM_SIZE( MPI_COMM_WORLD, isize, ierr)
      if( isize .ne. lnpx*lnpy*lnpz ) then
         write(*,*) 'ERROR: isize must match with lnpx*lnpy*lnpz.', &
                    isize, lnpx, lnpy, lnpz
         goto 999
      end if
      call MPI_COMM_RANK( MPI_COMM_WORLD, lrank, ierr)
!....    
      lrkx  = mod( lrank, lnpx )
      lrkxp = lrank + 1
      lrkxm = lrank - 1
      llbx  = llx * lrkx + 1
      lubx  = llbx + llx - 1 
      lrky  = mod( lrank, lnpx*lnpy ) / lnpx
      lrkyp = lrank + lnpx
      lrkym = lrank - lnpx
      llby  = lly * lrky + 1
      luby  = llby + lly - 1 
      lrkz  = lrank / ( lnpx * lnpy )
      lrkzp = lrank + ( lnpx * lnpy )
      lrkzm = lrank - ( lnpx * lnpy )
      llbz  = llz * lrkz + 1
      lubz  = llbz + llz - 1
!....
      call measure ( 1, 'total', 1 )
      call measure ( 1, 'adv3dx', 2 )
      call measure ( 1, 'adv3dy', 3 )
      call measure ( 1, 'adv3dz', 4 )
      call measure ( 1, 'cmp3dp', 5 )
      call measure ( 1, 'cmpram', 6 )
      call measure ( 1, 'reflect in x', 7 )
      call measure ( 1, 'reflect in y', 8 )
      call measure ( 1, 'reflect in z', 9 )
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
      wcheckp = 0.0d0
      izs = max( lz/4 + 1, llbz )
      izs = izs - llbz + 1
      ize = min( lz/4 * 3, lubz )
      ize = ize - llbz + 1
      iys = max( ly/4 + 1, llby )
      iys = iys - llby + 1
      iye = min( ly/4 * 3, luby )
      iye = iye - llby + 1
      ixs = max( lx/4 + 1, llbx )
      ixs = ixs - llbx + 1
      ixe = min( lx/4 * 3, lubx )
      ixe = ixe - llbx + 1
!..
      do iz = izs, ize
      do iy = iys, iye
      do ix = ixs, ixe
         wcheckp = wcheckp + sr(ix,iy,iz)
      end do
      end do
      end do
!.
      call MPI_REDUCE( wcheckp, wcheck, 1, MPI_DOUBLE_PRECISION, &
                       MPI_SUM, 0, MPI_COMM_WORLD, ierr )
!..
      if( lrank .eq. 0 ) then
        write(*,*) '--------------------------------------------------'
        write(*,*) 'mpi3h, lnpx, lnpy, lnpz = ', lnpx, lnpy, lnpz
!$      write(*,*) 'threads    = ', OMP_GET_MAX_THREADS()
        write(*,*) 'lx, ly, lz = ', lx, ly, lz
        write(*,*) 'lstep = ', lstep
        write(*,*) 'wtime, wcheck = ', wtime, wcheck
        call measure ( 0, ' ', 0 )
      end if
!....
  999 continue
      call MPI_FINALIZE( ierr )
!....
      stop
      end
