  subroutine sub1(a, b, n1, n2, n3)
    implicit none
    real(8), intent(in) :: a(n1, n2)
    real(8), intent(inout) :: b(n3)
    integer n1, n2, n3
    integer i, j, l
  end subroutine sub1



