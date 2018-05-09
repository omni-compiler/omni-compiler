module issue321

type t
end type t

interface operator(==)
  module procedure equal1
end interface

interface assignment(=)
 module procedure assign1
end interface


contains
  subroutine assign1(lhs, rhs)
    type(t), intent(inout) :: lhs
    integer, intent(in) :: rhs
  end subroutine assign1

  logical function equal1(lhs, rhs)
    type(t), intent(in) :: lhs
    type(t), intent(in) :: rhs
    equal1 = .TRUE.
  end function equal1

end module issue321
