module mod1
contains
  subroutine sub1(t1, c1)
    type(*), intent(in) :: t1
    class(*), intent(in) :: c1
  end subroutine sub1
end module mod1
