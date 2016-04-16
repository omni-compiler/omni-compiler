module mmm
!!   include "xmp_coarray.h"
end module mmm

use mmm
real dada(10)[*]
do i=1,10; dada(i)=this_image()**i; enddo
syncall
x = dada(3)[2]
if (abs(x-2.0**3)<0.00001) then
   print '("[",i0,"] OK")', this_image()
else
   print '("[",i0,"] NG: x should be 2.0**3 but",f10.2)', this_image(), x
endif
end
