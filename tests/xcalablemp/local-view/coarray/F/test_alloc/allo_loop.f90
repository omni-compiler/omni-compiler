  program times2
    include "xmp_lib.h"
    real, allocatable :: abc(:,:)[:]

    do i=1,10000
       allocate(abc(1000,1000)[*])
       deallocate(abc)
    end do

  end program times2
