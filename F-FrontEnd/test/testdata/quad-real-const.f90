program main

  integer, parameter :: ks = kind(1.0e0), kd = kind(1.0d0), kq = kind(1.0q0)
  real(8) :: a = 2.0d0
  real :: b = 2.0e0
  real(16) :: c = 2.0
  real :: d = 2.0q0
  real(kq) :: e = 2.0_kq

  a = sqrt(a)
  b = sqrt(b)
  c = sqrt(c)
  d = sqrt(d)
  e = sqrt(e)

  print *, ks, kd, kq
  print *, a, b, c, d
  print *, e

end program main

