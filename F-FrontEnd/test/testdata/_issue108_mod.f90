module issue108_mod
  implicit none
  integer, parameter :: ps = 6
  integer, parameter :: rs = 37
  integer, parameter :: pd = 12
  integer, parameter :: rd = 307
  integer, parameter :: pi4 = 9
  integer, parameter :: pi8 = 14
  integer, parameter :: sp = selected_real_kind(ps,rs)
  integer, parameter :: dp = selected_real_kind(pd,rd)
  integer, parameter :: wp = dp
  integer, parameter :: i4 = selected_int_kind(pi4)
  integer, parameter :: i8 = selected_int_kind(pi8)
end module issue108_mod
