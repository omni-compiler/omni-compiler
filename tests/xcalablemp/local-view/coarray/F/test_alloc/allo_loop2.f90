  program times2
    include "xmp_lib.h"
    real, allocatable :: abc(:,:)[:]
    me = this_image()

    do i=1,10000
       syncall
       if (mod(i,10)==0.and.me==2) then
          call xmpf_coarray_msg(1)
       endif
       allocate(abc(1000,1000)[*])
       syncall
       if (mod(i,10)==0) then
          write(*,100) i, xmpf_coarray_allocated(),  xmpf_coarray_suspended()
       endif
       syncall
       deallocate(abc)
       syncall
       if (mod(i,10)==0.and.me==2) then
          call xmpf_coarray_msg(0)
       endif
    end do

100 format("i=",i0,": allocated ",i0," bytes, suspended ",i0," bytes")

  end program times2
