  program a0d0
    include "xmp_coarray.h"
    real a0[*]
    integer xmp_node_num
    real data0, data1

    me = xmp_node_num()   ! == this_image()

    !---------------------------- init
    data0 = 7.77
    data1 = 1.2
    a0 = data0
    sync all

    !---------------------------- exec
    if (me==1) then
       a0[2]=1.2
    end if

    sync all

    !---------------------------- check and output start
    if (me==2) then
       if (a0==data1) then
          write(*,101) me
       else
          write(*,100) me
       endif
    else
       if (a0==data0) then
          write(*,101) me
       else
          write(*,100) me
       endif
    end if

100 format ("[",i0,"] NG")
101 format ("[",i0,"] OK")

  end program
