program test
  integer a(100)
  block
  do concurrent (i=1:m)
     a(i) = a(i)
  end do
  end block
end program test
