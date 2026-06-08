program test

  use :: flcl_util_kokkos_mod
!  use :: flcl_mod
  
  integer, parameter :: N1 = 4096
  integer, parameter :: N2 = 4096

  real :: a
  real :: x(N1,N2), y(N1,N2)

  a = 0.5e0

  x = 1.23e0
  y = 4.56e0

  call kokkos_initialize()
  call sub(a, x, y)
  call kokkos_finalize()

  print *, y(128,128)

end program test

