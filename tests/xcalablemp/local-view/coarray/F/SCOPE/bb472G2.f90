module mmm
  !$xmp nodes p(3)
  !$xmp template t(10)
  !$xmp distribute t(block) onto p
  real a(10,10)
  !$xmp align a(*,i) with t(i)
contains
  subroutine mmmm
    !$xmp reflect (a)
  end subroutine mmmm
end module mmm

use mmm
call mmmm
print *,"OK"
end
