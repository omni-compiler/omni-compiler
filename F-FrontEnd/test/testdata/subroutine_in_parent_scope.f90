module mmm

  implicit none
  public :: sub1, sub2

contains

  subroutine sub1
    call sub2
    call sub2
  end subroutine sub1

  subroutine sub2
  end subroutine sub2

end module mmm
