program main
  use common
  character*15 :: arg1
  integer*8 :: n, n_specified
  integer*8 :: loc_N, HPLMaxProcMem
  integer :: n_nodes, xmp_num_nodes
  integer :: n_threads, omp_get_max_threads

  call getarg(1, arg1)
  read (arg1, "(I15)") n_specified

  n_nodes = xmp_num_nodes()
  n_threads = omp_get_max_threads() 

  !! number of processes have been factored out - need to put it back in
  HPLMaxProcMem = n_specified / n_nodes * 64
  loc_N = HPLMaxProcMem / (4*n_nodes) / 16
  n = loc_N * n_nodes * n_nodes

  call xmpfft(n, n_nodes, n_threads)

end program
