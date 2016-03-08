module mmm
  include "xmp_coarray.h"
end module mmm

use mmm
real dada(10)[*]
x = dada(3)[2]
end
