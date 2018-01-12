program test
  integer a(100)
  do concurrent (i=1:m)
     a(k+i) = a(k+i) + factor*a(l+i)
  end do
end program test

