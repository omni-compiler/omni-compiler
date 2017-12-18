module mod1
use issue285, only: ngpgfz, c8
integer, parameter :: c4 = 4
contains

  subroutine sub1(i)
    integer :: i

  end subroutine sub1

  subroutine sub2()

    call sub1(ngpgfz)
    call sub1(c8)
    call sub1(c4)

  end subroutine sub2

end module mod1
