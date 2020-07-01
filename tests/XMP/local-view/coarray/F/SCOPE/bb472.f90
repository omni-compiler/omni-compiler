module bb
!!   include "xmp_coarray.h"
  real x
contains
  subroutine bbm
    real dada(10)[*]
    do i=1,10; dada(i)=this_image()**i
    enddo
    syncall
    x = dada(3)[2]
  end subroutine bbm
end module bb

use bb
call bbm
if (abs(x-2.0**3)<0.00001) then
   print '("[",i0,"] OK")', this_image()
else
   print '("[",i0,"] NG: x should be 2.0**3 but",f10.2)', this_image(), x
   call exit(1)
endif
end
