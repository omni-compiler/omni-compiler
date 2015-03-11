  program decl
    include "xmp_lib.h"
!!    real, allocatable :: a(:)[:]   #390
    real, allocatable :: a(:)[*]

  end program decl
