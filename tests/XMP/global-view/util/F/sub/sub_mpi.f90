subroutine sub_mpi(comm)
  include 'mpif.h'
  integer comm, rank, irank

  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_rank(comm, irank, ierr)

  if(rank == 2) then
    if(irank == 1) then
      print *,"PASS"
    else
      print *, "ERROR rank=",irank
      call exit(1)
    end if
  end if

end subroutine
