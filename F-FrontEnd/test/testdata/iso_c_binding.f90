module mod1

type, bind(C) :: ctype
  integer :: len
end type ctype

interface
  function send_to_port(port, data, length) bind(C)
    use, intrinsic :: iso_c_binding, only: c_int, c_char
    implicit none
    integer(c_int), value :: port
    character(kind=c_char) :: data(*)
    integer(c_int), value :: length
    integer(c_int) :: send_to_port
  end function send_to_port
end interface


contains
  function sleep() bind(c, name="sleep")
  end function sleep

  subroutine dummy() bind(c, name="dummySleep")
  end subroutine dummy

end module mod1
