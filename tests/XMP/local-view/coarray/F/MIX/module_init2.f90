  module mom2
!!     include "xmp_coarray.h"
    !$xmp nodes p(4)
    !$xmp template t(10)
    !$xmp distribute t(block) onto p
    real a(10,10)
    !$xmp align a(*,i) with t(i)
    real z(10,10)[*]
  end module mom2

  subroutine sas
    use mom2
    !$xmp nodes pp(4)
    !$xmp template tt(10)
    !$xmp distribute tt(cyclic) onto pp
    real aa(10,10)
    !$xmp align aa(*,i) with tt(i)
  end subroutine sas
