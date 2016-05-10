!!   include "xmp_coarray.h"
  integer,save :: itmp(3)[*]

  itmp(1)=3
  if (size(itmp)==3.and.itmp(1)==3) then
     write(*,*) "OK"
  else 
     write(*,*) "NG: itmp(3)[*] is illegally allocated or not allocated."
  endif
  end
