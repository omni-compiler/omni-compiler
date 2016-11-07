module mod1
use, intrinsic :: iso_c_binding

type, bind(C) :: ctype
  integer :: len
end type ctype

interface
  function send_to_port(port, data, length) bind(C)
    implicit none
    integer(C_INT), value :: port
    character(kind=C_CHAR) :: data(*)
    integer(C_INT), value :: length
    integer(C_INT) :: send_to_port
  end function send_to_port
end interface


contains
  function sleep() bind(C, name="sleep")
  end function sleep

  subroutine dummy() bind(C, name="dummySleep")
  end subroutine dummy

end module mod1
