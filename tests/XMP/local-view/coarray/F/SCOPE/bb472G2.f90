module bb2
  !$xmp nodes p(3)
  !$xmp template t(10)
  !$xmp distribute t(block) onto p
  real a(10,10)
  !$xmp align a(*,i) with t(i)
contains
  subroutine bb2m
    !$xmp reflect (a)
  end subroutine bb2m
end module bb2

use bb2
call bb2m
print *,"OK"
end
