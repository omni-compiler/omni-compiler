  include "xmp_coarray.h"
  integer zz(3)[*] = (/1111,2222,3333/)
  integer yy(3)

  if (this_image()==1) then
     yy = zz(1:3:1)[3]
  endif

  syncall

  write(*,*) "yy=",yy

  end
