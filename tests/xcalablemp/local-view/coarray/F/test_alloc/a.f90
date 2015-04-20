  include "xmp_coarray.h"
  real, allocatable :: aaa(:)[:]

      logical :: res
      res = allocated(aaa)
  end
