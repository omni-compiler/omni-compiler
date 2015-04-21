program main
  include "xmp_coarray.h"
  integer a1d1(10)[*]
  integer tmp(5)

  n=6
  tmp = a1d1(2:n)[3]

end program

