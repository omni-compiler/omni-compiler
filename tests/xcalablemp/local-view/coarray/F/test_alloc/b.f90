  include "xmp_lib.h"
  real, allocatable :: aaa(:)[:]

  contains
    subroutine ppp
      logical :: res
      res = allocated(aaa)
    end subroutine ppp
  end
