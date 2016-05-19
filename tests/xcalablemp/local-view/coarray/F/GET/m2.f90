  module mmm2
!!     include "xmp_coarray.h"
    real*8 da(10,10)[3,*]

  end module mmm2

  use mmm2

  da(3,:) = da(:,5)[3,1]

 end

