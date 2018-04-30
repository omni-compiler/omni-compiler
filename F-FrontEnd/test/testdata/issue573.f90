module mod1
implicit none
integer, parameter :: wp = selected_real_kind(13,7)

contains

  real(wp) elemental function q_effective(qsat, fsat)
     real(wp), intent(in) :: qsat, fsat
     q_effective = qsat * fsat
  end function q_effective

  subroutine sub1()
    real(wp) :: q_air_eff(10)
    real(wp) :: q_air(10)
    integer :: nc
    nc = 10

    q_air_eff(:) = q_effective(SPREAD(0._wp, NCOPIES=nc, DIM=1), SPREAD(1._wp, NCOPIES=nc, DIM=1))
  end subroutine sub1

end module mod1

