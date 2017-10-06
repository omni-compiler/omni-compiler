module mod1
  type t 
    contains
      procedure, nopass, public :: f
      procedure, nopass, public :: g
      ! Should be the same result
      ! generic :: p => f, g
      generic :: p => f
      generic :: p => g
  end type t
contains
  subroutine f(i)
    integer :: i
  end subroutine f

  subroutine g(i)
    real :: i
  end subroutine g
end module mod1

