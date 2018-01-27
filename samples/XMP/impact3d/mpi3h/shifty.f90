!-----------------------------------------------------------------
! shift communication in y
!
      subroutine shifty ( fa, kn, kx, ky, kz, kwidth )
!
      use parameter
      use constant
      include "implicit.h"
      dimension fa(kn,kx,ky,kz), ireq(2), ist(MPI_STATUS_SIZE,2)
      save    
!....
      if ( lnpy .eq. 1 ) return
!....
      icount = kn * kx * kz * abs(kwidth)
      ireqn = 0
!....
      if ( kwidth .gt. 0 ) then
        if ( lrky .ne. lnpy-1 ) then 
          icn = 0
	  do iz = 1, kz
	  do iy = ky+1-kwidth, ky
	  do ix = 1, kx
	  do in = 1, kn
             icn = icn + 1
	     sseny(icn) = fa(in,ix,iy,iz)
	  end do
	  end do
	  end do
	  end do
          ireqn = ireqn + 1
          call MPI_ISEND( sseny, icount, &
                          MPI_DOUBLE_PRECISION, lrkyp, &
                          0, MPI_COMM_WORLD, ireq(ireqn), ierr )
        end if
!..
        if ( lrky .ne. 0 ) then
          ireqn = ireqn + 1
          call MPI_IRECV( srecy, icount, &
                          MPI_DOUBLE_PRECISION, lrkym, &
                          0, MPI_COMM_WORLD, ireq(ireqn), ierr )
        end if
      else if ( kwidth .lt. 0 ) then
        if ( lrky .ne. 0 ) then
          icn = 0
	  do iz = 1, kz
	  do iy = 1, -kwidth
	  do ix = 1, kx
	  do in = 1, kn
             icn = icn + 1
	     sseny(icn) = fa(in,ix,iy,iz)
	  end do
	  end do
	  end do
	  end do
          ireqn = ireqn + 1
          call MPI_ISEND( sseny, icount, &
                          MPI_DOUBLE_PRECISION, lrkym, &
                          1, MPI_COMM_WORLD, ireq(ireqn), ierr )
        end if
!..
        if ( lrky .ne. lnpy-1 ) then
          ireqn = ireqn + 1
          call MPI_IRECV( srecy, icount, &
                        MPI_DOUBLE_PRECISION, lrkyp, &
                        1, MPI_COMM_WORLD, ireq(ireqn), ierr )
        end if
      end if
!....
      call MPI_WAITALL( ireqn, ireq, ist, ierr )
!..
      if ( kwidth .gt. 0 ) then
          icn = 0
	  do iz = 1, kz
	  do iy = 1, kwidth
	  do ix = 1, kx
	  do in = 1, kn
             icn = icn + 1
	     fa(in,ix,iy,iz) = srecy(icn)
	  end do
	  end do
	  end do
	  end do
      else if ( kwidth .lt. 0 ) then
          icn = 0
	  do iz = 1, kz
	  do iy = ky+kwidth+1, ky
	  do ix = 1, kx
	  do in = 1, kn
             icn = icn + 1
	     fa(in,ix,iy,iz) = srecy(icn)
	  end do
	  end do
	  end do
	  end do
      end if
!....
      return
      end
