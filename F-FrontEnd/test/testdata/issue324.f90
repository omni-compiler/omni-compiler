module mod1
  implicit none
  
  type typ1
  end type typ1

contains

  subroutine sub1(rhs)
    class(typ1), optional :: rhs
  end subroutine sub1

end module mod1
