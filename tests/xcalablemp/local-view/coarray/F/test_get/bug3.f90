  include "xmp_coarray.h"
  integer a1d1(10)[*]
  
  me = xmp_node_num()
  do i=1,10
     a1d1(i)=i*me
  enddo

  if (me==1) then
     a1d1(2:a1d1(3)[2])[3]=-100
  endif
  
  write(*,*) me,a1d1
  end

