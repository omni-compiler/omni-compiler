  include "xmp_coarray.h"
  type new_t
     integer n1, n2
  end type new_t
  type(new_t) :: bebe
  type(new_t) :: meeta[*]
  write(*,*) meeta[2]
  end
