module mod1
use, intrinsic :: iso_c_binding

type, bind(C) :: ctype
  integer :: len
end type ctype

interface
  function send_to_port(port, data, length) bind(C)
    use, intrinsic :: iso_c_binding
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

  function fct5() result(res)
    integer :: res
    res = 1
  end function fct5

  function fct1() result(res) bind(C, name='c_fct1')
    integer :: res
    res = 1
  end function

  function fct2() bind(C, name='c_fct2') result(res)
    integer :: res
    res = 1
  end function

  integer function fct3() bind(C, name='c_fct3') result(res)
    res = 1
  end function

  integer function fct4() result(res) bind(C, name='c_fct4')
    res = 1
  end function


end module mod1
