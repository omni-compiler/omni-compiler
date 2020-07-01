!!module m_m2
  !$xmp nodes p(3)
  !$xmp template t(10)
  !$xmp distribute t(block) onto p
  real a(10,10)
  !$xmp align a(*,i) with t(i)
!!end module m_m2

!!use m_m2
!$xmp reflect (a)
end 

