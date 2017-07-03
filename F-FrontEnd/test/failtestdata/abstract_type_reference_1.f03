module m
  implicit none
  type, abstract :: t
    integer :: i
  end type
  type, extends(t) :: tt
    class(t), pointer :: comp
  end type

  type(tt), target :: c1
  class(tt), allocatable :: c2
  class(t), pointer :: cp

contains
  subroutine sub(arg)
    c1%t = c1%t
  end subroutine
end module m
