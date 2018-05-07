!!   include "xmp_coarray.h"
  integer a(20)[*]

  me = xmp_node_num()
  do i=1,20
     a(i) = i*me
  enddo
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all

  n1 = a(10)[1]
  n2 = a(10)[2]
  n3 = a(10)[3]
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  sync all
  write(*,*) "me,n1,n2,n3=",me,n1,n2,n3
  end
