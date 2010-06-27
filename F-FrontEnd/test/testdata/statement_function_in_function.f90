integer function f(a, b)
  integer a, b, x, y
  g(x, y) = x ** y + 1
  f = g(a, b)
end function

program main
  integer f
  print *, f(3,3)
end program
