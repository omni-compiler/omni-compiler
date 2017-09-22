module mo_constant
  implicit none

  integer, parameter :: wp = selected_real_kind(13)
  real(kind=wp), parameter :: c0 = 0._wp
end module mo_constant
