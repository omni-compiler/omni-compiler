!!   include "xmp_coarray.h"
  integer a1d1(10)[*]
  integer tmp(5)

  tmp = a1d1(2:1+5)[3]
  sync all

end program

