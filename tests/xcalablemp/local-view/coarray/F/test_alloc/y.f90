program allo2
  include "xmp_lib.h"
  real, allocatable :: a(:)[:] 

  call allo

    allocate (a(10))
      
      
end program allo2
