module mod1
  type t 
    contains
      procedure, nopass, public :: f
      procedure, nopass, public :: g
      procedure, nopass, public :: h
      ! Should be the same result
      ! generic :: p => f, g, h
      generic :: p => f
      generic :: p => g
      generic :: p => h
  end type t
contains
  subroutine f(i)
    integer :: i
  end subroutine f

  subroutine g(i)
    real :: i
  end subroutine g

  subroutine h(i)
    logical :: i
  end subroutine h
end module mod1

