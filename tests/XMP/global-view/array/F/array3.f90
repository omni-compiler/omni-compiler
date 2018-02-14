program test

  !$xmp nodes p(2,2)
  !$xmp template t(8,8,8)
  !$xmp distribute t(*,block,block) onto p

  real a(8,8,8)
  !$xmp align a(*,j,k) with t(*,j,k)

  !$xmp array on t
  a = 0.

!$xmp task on p(1,1)
  write(*,*) "PASS"
!$xmp end task

end program test
