module mod1
  use mo_random
implicit none
contains
  subroutine sub1
    implicit none
    real :: z(2)
    call random_number(z)
  end subroutine sub1
end module mod1
