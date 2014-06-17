!$xmp nodes p(*)
!$xmp template t(100)
!$xmp distribute t(block) onto p

  real a0(100), b0(100), c0(100)
!$xmp align (i) with t(i) :: a0, b0, c0

  real a(100), b(100), c(100)
!$xmp align (i) with t(i) :: a, b, c

  real, parameter :: PI = 3.14159265359
  integer :: result = 0

!$xmp loop on t(i)
  do i = 1, 100
     a0(i) = real(i) * (2. * PI / 100.)
     b0(i) = real(i) * (2. * PI / 100.)
     a(i)  = real(i) * (2. * PI / 100.)
     b(i)  = real(i) * (2. * PI / 100.)
  end do

!$xmp loop on t(i)
  do i = 1, 100
     c0(i) = sin(a0(i) + b0(i))
  end do

!$xmp array on t
  c = sin(a + b)

!$xmp loop on t(i)
  do i = 1, 100
     if (c0(i) /= c(i)) result = -1
  end do

!$xmp reduction(+:result)

!$xmp task on p(1)
  if ( result .eq. 0 ) then
     write(*,*) "PASS"
  else
     write(*,*) "ERROR"
     call exit(1)
  endif
!$xmp end task

end program
