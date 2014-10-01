program main
  integer rank, irank, isize, ierr
  integer xmp_get_mpi_comm
  integer comm

!$xmp nodes p(4)

  call xmp_init_mpi()

!$xmp task on p(2:3)
  comm = xmp_get_mpi_comm()
  call sub_mpi(comm)
!$xmp end task

  call xmp_finalize_mpi()

end program
