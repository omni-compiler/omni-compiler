  subroutine subsub
    include "xmp_coarray.h"
    real, allocatable :: aaa(:)[:]

  contains
    subroutine ppp
      logical :: res
      res = allocated(aaa)
    end subroutine ppp
  end subroutine subsub
