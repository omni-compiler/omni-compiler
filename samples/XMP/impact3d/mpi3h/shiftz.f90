!-----------------------------------------------------------------
! shift communication in z
!
      subroutine shiftz ( fa, kn, kx, ky, kz, kwidth )
!
      use parameter
      use constant
      include "implicit.h"
      dimension fa(kn,kx,ky,kz), ireq(2), ist(MPI_STATUS_SIZE,2)
      save
!....
      if ( lnpz .eq. 1 ) return
!....
      icount = kn * kx * ky * abs(kwidth)
      ireqn = 0
!....
      if ( kwidth .gt. 0 ) then
        if ( lrkz .ne. lnpz-1 ) then 
          ireqn = ireqn + 1
          call MPI_ISEND( fa(1,1,1,kz-kwidth+1), icount, &
                          MPI_DOUBLE_PRECISION, lrkzp, &
                          0, MPI_COMM_WORLD, ireq(ireqn), ierr )
        end if
!..
        if ( lrkz .ne. 0 ) then
          ireqn = ireqn + 1
          call MPI_IRECV( fa(1,1,1,1), icount, &
                          MPI_DOUBLE_PRECISION, lrkzm, &
                          0, MPI_COMM_WORLD, ireq(ireqn), ierr )
        end if
      else if ( kwidth .lt. 0 ) then
        if ( lrkz .ne. 0 ) then
          ireqn = ireqn + 1
          call MPI_ISEND( fa(1,1,1,1), icount, &
                          MPI_DOUBLE_PRECISION, lrkzm, &
                          1, MPI_COMM_WORLD, ireq(ireqn), ierr )
        end if
!..
        if ( lrkz .ne. lnpz-1 ) then
          ireqn = ireqn + 1
          call MPI_IRECV( fa(1,1,1,kz+kwidth+1), icount, &
                          MPI_DOUBLE_PRECISION, lrkzp, &
                          1, MPI_COMM_WORLD, ireq(ireqn), ierr )
        end if
      end if
!....
      call MPI_WAITALL( ireqn, ireq, ist, ierr )
!....
      return
      end
