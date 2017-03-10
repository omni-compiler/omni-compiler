program prog1
  integer, parameter :: dp = selected_real_kind(12,37)
  integer, parameter :: wp = dp
  integer, parameter :: ii = selected_int_kind(8)

  integer(ii) :: i1 = 10_ii
  real(wp) :: r1 = 10.0_wp
  real(wp) :: ki = 23._wp
  real(wp) :: r2 = 1.380650e-23_wp

end program prog1
