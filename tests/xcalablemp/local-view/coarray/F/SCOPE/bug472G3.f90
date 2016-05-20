module m_m3
  !$xmp nodes p(4)
  !$xmp template t(10)
  !$xmp distribute t(block) onto p
  real a(10,10)
  !$xmp align a(*,i) with t(i)
end module m_m3

subroutine zzz
use m_m3
!$xmp reflect (a)
end subroutine zzz

