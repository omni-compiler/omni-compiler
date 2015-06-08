  program decl
    include "xmp_coarray.h"
    real r1(10)[4,*]
    real r2(:)[:,:]
    allocatable r2

  end program decl
