  include "xmp_coarray.h"
  real, allocatable :: a(:,:)[:,:,:]
  allocate (a(1:2,3:5)[6:9,8:10,-3:*])
  nnnn = image_index(a, [8,9,3])
  write(*,*) nnnn, 2+4*1+4*3*6
  end
