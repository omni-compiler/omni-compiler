  include "xmp_coarray.h"
  real, allocatable :: buf3u(:,:)[:]

  allocate(buf3u(2:8, 2:8)[*])
  write(*,*) lbound(buf3u,1), ubound(buf3u,1), lbound(buf3u,2), ubound(buf3u,2)

  end

