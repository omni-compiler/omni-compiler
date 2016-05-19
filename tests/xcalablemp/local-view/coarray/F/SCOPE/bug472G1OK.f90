!!module m_m1
  !$xmp nodes p(3)
  !$xmp template t(10)
  !$xmp distribute t(block) onto p
!!end module m_m1

!!use m_m1
real a(10,10)
!$xmp align a(*,i) with t(i)
end 

