program allo2
  include "xmp_coarray.h"
  real, allocatable :: a(:)[:] 

  call allo

    allocate (a(10))
      
      
end program allo2
