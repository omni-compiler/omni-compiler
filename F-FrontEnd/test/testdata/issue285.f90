module mod1
use issue285, only: ngpgfz
contains

  subroutine sub1(i)
    integer :: i

  end subroutine sub1

  subroutine sub2()

    call sub1(ngpgfz)

  end subroutine sub2

end module mod1
