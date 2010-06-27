subroutine f(a, b)
  integer a, b, x, y
  g(x, y) = x ** y + 1
  print *, g(a, b)
end subroutine

program main
  call f(3,3)
end program
