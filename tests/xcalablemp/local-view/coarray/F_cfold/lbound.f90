!! execute with eivironment XMPF_COARRAY_MSG=1

  include "xmp_coarray.h"
  real aaa(2:101,3)
  real bb1(lbound(aaa,1))[*]     !! size=2
  real bb2(ubound(aaa,2))[*]     !! size=3

  end
