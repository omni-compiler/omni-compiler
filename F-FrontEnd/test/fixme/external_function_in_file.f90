function f()
  integer :: f
  f = 10
end function f

program main
  implicit none!!!!!!!!
  integer a
  a = f()
end program main
