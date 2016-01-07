  include "xmp_coarray.h"
  real :: a(1:2,3:5)[6:*]
  nnnn = image_index(a, [8])
  write(*,*) nnnn, 3
  end
