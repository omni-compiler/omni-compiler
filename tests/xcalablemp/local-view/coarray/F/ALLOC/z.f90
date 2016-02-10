program allo2
  include "xmp_coarray.h"
  real, allocatable :: a(:)[:] 

  call allo

contains
  subroutine allo
    allocate (a(10))
  end subroutine allo
      
      
end program allo2
