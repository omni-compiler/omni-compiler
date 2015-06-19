subroutine sub(V3)
  include "xmp_coarray.h"
  integer n(10)
  integer :: V3(:)[:,:]      !! not allowed in F2008: implicit size

  !! explicit size
  n(1:5) = V3(1:5)[k1,k2]

  !! deferred sizes
  n(:) = V3(:)[k1,k2]
  n(1:5) = V3(:5)[k1,k2]
  n(1:5) = V3(6:)[k1,k2]
  n(1:5) = V3(::2)[k1,k2]
  n(1:5) = V3(2::2)[k1,k2]
  n(1:4) = V3(:8:2)[k1,k2]

end subroutine sub

end
