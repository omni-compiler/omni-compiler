  include "xmp_coarray.h"
  integer a1d1(10)[*]
  integer tmp(5)

  tmp = a1d1(2:ifoo())[3]
  sync all

end program

integer function ifoo()
  ifoo=6
end function ifoo
