module mod1
  use, intrinsic :: iso_c_binding, only: c_int
  integer(c_int), bind(c,name='NO_OF_SEC_IN_A_DAY') :: no_of_sec_in_a_day
  integer(c_int), bind(c,name='NO_OF_SEC_IN_A_HOUR') :: no_of_sec_in_a_hour
end module mod1