  module xx
!!     include "xmp_coarray.h"
    integer,save:: aaa[*]
  end module xx

  subroutine zz(c)
    use xx
    integer,save:: bbb[*]

    return
  end subroutine zz

