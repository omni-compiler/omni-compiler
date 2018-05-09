program test
  INTEGER, PARAMETER, DIMENSION(3) :: arr = (/1,2,3/)
  error stop (123 + 456)
  error stop arr(1)
  error stop 'foo' // 'bar'
end program test
