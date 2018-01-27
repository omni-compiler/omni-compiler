program test
  integer a(100)
  xxxx: do concurrent (i=1:m)
     a(k+i) = a(k+i) + factor*a(l+i)
  end do xxxx
end program test
