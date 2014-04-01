module mod0
contains
  subroutine sub0(n)
    integer, intent(in) :: n
  end subroutine sub0
end module mod0

subroutine foo
  use mod0
end subroutine foo
