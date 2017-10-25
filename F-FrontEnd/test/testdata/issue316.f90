module mod1
implicit none
private
integer, parameter :: p1 = 4

interface
  subroutine sub1()
  end subroutine
end interface

public :: sub1
end module mod1
