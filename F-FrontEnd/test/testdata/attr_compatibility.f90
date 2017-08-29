module mod1

  public :: typ1

  type, bind(C) :: typ1
    integer :: i1
  end type typ1

  private :: typ2

  type, bind(C) :: typ2
    integer :: i2
  end type typ2

end module mod1
