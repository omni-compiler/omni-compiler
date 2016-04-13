module mmm
  !$xmp nodes p(3)
  !$xmp template t(10)
  !$xmp distribute t(block) onto p
end module mmm

use mmm
real a(10,10)
!$xmp align a(*,i) with t(i)
print *,"OK"
end 

