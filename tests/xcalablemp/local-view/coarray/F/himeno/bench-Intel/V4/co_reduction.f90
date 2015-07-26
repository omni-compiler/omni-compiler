subroutine co_sum(source, result)
  include "mpif.h"
  real(kind=4) :: source, result
  integer ierr

  call mpi_allreduce(source, result, 1, mpi_real4, &
       mpi_sum, mpi_comm_world, ierr)

  if (ierr == 0) then
     call mpi_barrier(mpi_comm_world, ierr)
  end if

  if (ierr /= 0) then
     stop 111
  end if
  return
end subroutine


subroutine co_max(source, result)
  include "mpif.h"
  real(kind=8) :: source, result
  integer ierr

  call mpi_allreduce(source, result, 1, mpi_real8, &
       mpi_max, mpi_comm_world, ierr)

  if (ierr == 0) then
     call mpi_barrier(mpi_comm_world, ierr)
  end if

  if (ierr /= 0) then
     stop 222
  end if
  return
end subroutine

