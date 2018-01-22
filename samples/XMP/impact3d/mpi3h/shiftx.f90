!-----------------------------------------------------------------
! shift communication in x
!
      subroutine shiftx ( fa, kn, kx, ky, kz, kwidth )
!
      use parameter
      use constant
      include "implicit.h"
      dimension fa(kn,kx,ky,kz), ireq(2), ist(MPI_STATUS_SIZE,2)
      save    
!....
      if ( lnpx .eq. 1 ) return
!....
      icount = kn * ky * kz * abs(kwidth)
      ireqn = 0
!....
      if ( kwidth .gt. 0 ) then
        if ( lrkx .ne. lnpx-1 ) then 
          icn = 0
	  do iz = 1, kz
	  do iy = 1, ky
	  do ix = kx+1-kwidth, kx
	  do in = 1, kn
             icn = icn + 1
	     ssenx(icn) = fa(in,ix,iy,iz)
	  end do
	  end do
	  end do
	  end do
          ireqn = ireqn + 1
          call MPI_ISEND( ssenx, icount, &
                          MPI_DOUBLE_PRECISION, lrkxp, &
                          0, MPI_COMM_WORLD, ireq(ireqn), ierr )
        end if
!..
        if ( lrkx .ne. 0 ) then
          ireqn = ireqn + 1
          call MPI_IRECV( srecx, icount, &
                          MPI_DOUBLE_PRECISION, lrkxm, &
                          0, MPI_COMM_WORLD, ireq(ireqn), ierr )
        end if
      else if ( kwidth .lt. 0 ) then
        if ( lrkx .ne. 0 ) then
          icn = 0
	  do iz = 1, kz
	  do iy = 1, ky
	  do ix = 1, -kwidth
	  do in = 1, kn
             icn = icn + 1
	     ssenx(icn) = fa(in,ix,iy,iz)
	  end do
	  end do
	  end do
	  end do
          ireqn = ireqn + 1
          call MPI_ISEND( ssenx, icount, &
                          MPI_DOUBLE_PRECISION, lrkxm, &
                          1, MPI_COMM_WORLD, ireq(ireqn), ierr )
        end if
!..
        if ( lrkx .ne. lnpx-1 ) then
          ireqn = ireqn + 1
          call MPI_IRECV( srecx, icount, &
                        MPI_DOUBLE_PRECISION, lrkxp, &
                        1, MPI_COMM_WORLD, ireq(ireqn), ierr )
        end if
      end if
!....
      call MPI_WAITALL( ireqn, ireq, ist, ierr )
!..
      if ( kwidth .gt. 0 ) then
          icn = 0
	  do iz = 1, kz
	  do iy = 1, ky
	  do ix = 1, kwidth
	  do in = 1, kn
             icn = icn + 1
	     fa(in,ix,iy,iz) = srecx(icn)
	  end do
	  end do
	  end do
	  end do
      else if ( kwidth .lt. 0 ) then
          icn = 0
	  do iz = 1, kz
	  do iy = 1, ky
	  do ix = kx+kwidth+1, kx
	  do in = 1, kn
             icn = icn + 1
	     fa(in,ix,iy,iz) = srecx(icn)
	  end do
	  end do
	  end do
	  end do
      end if
!....
      return
      end
