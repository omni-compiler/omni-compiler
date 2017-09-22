module mod1
  implicit none
  use mo_constant
contains
  subroutine sub1()
    implicit none
    real(kind=wp) :: gcc(4) = (/c0, 500._wp, c0, c0/)
  end subroutine sub1
end module mod1
