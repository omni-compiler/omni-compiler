module mod1

type, bind(C) :: ctype
  integer :: len
end type ctype

contains
  function sleep() bind(c, name="sleep")
  end function sleep

  subroutine dummy() bind(c, name="dummySleep")
  end subroutine dummy

end module mod1
