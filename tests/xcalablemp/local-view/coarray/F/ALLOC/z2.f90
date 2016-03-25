subroutine azz
    include "xmp_coarray.h"
    integer*8 a(100)[*]

    do i=1,100
       a(i) = i+me*100
    end do

    b = -99


  end subroutine azz
