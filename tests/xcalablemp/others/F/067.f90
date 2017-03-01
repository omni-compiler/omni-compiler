module parameter
  integer, parameter :: lx = 2048
  integer, parameter :: ly = 2048
end module parameter

subroutine foo
  use parameter
end subroutine foo
