  include "xmp_coarray.h"
  integer a1d1(10)[*]
  integer tmp(10)
  
  tmp = a1d1(2:a1d1(3)[2])[3]
  sync all

end program

