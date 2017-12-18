program test
  include 'mpif.h'
  integer ierror
  
  call MPI_INIT(ierror)
  call xmp_init(MPI_COMM_WORLD)
  call hoge()
  call xmp_finalize()
  call MPI_FINALIZE(ierror)
  
end program test
