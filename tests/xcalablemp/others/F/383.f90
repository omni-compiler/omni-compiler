program main
!$xmp nodes p(2)
!$xmp template t(10)
!$xmp distribute t(block) onto p

  real a(10)
!$xmp align a(i) with t(i)

  call foo(a)

contains

  subroutine foo(c)
    real c(0:)
!$xmp align c(i) with t(i+1)
  end subroutine foo

end program main
