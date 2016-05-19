  module mmm3
!!     include "xmp_coarray.h"
    real*8 da(10,10)[3,*]


  end module mmm3

  subroutine pape
    use mmm3

    sync all

    continue


  end subroutine pape


