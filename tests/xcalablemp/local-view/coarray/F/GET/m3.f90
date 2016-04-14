  module mmm
!!     include "xmp_coarray.h"
    real*8 da(10,10)[3,*]


  end module mmm

  subroutine pape
    use mmm

    sync all

    continue


  end subroutine pape


