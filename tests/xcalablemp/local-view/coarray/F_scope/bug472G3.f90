module mmm
  !$xmp nodes p(4)
  !$xmp template t(10)
  !$xmp distribute t(block) onto p
  real a(10,10)
  !$xmp align a(*,i) with t(i)
end module mmm

subroutine zzz
use mmm
!$xmp reflect (a)
end subroutine zzz

