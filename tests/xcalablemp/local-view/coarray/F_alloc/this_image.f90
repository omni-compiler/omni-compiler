  include "xmp_coarray.h"
  real a(1:2,3:5)[6:7,8:10,-3:*]
  n1 = this_image(a, 1)
  n2 = this_image(a, 2)
  n3 = this_image(a, 3)
  write(*,*) n1,n2,n3, "6-to-7, 8-to-9, -3"
  end
