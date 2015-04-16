subroutine sub(V3)
  include "xmp_lib.h"
  integer n(10)
  integer, allocatable :: V3(:)[:,:]

  !! deferred sizes
  n(:) = V3(:)[1,2]

end subroutine sub

end
