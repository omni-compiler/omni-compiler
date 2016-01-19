  program test_s
    include "xmp_coarray.h"
    complex*16 v[4,*], vv
!!    integer xmp_node_num
    integer nerr

    me = xmp_node_num()   ! == this_image()

    !---------------------------- switch on message
!!    call xmpf_coarray_msg(1)

    !---------------------------- initialization
    v = (1.23d4, 5.67d8)
    vv = v*v
    sync all

    !---------------------------- execution
    if (me==2) then
       v[1,1] = vv
    end if

    sync all

    !---------------------------- check and output start
    eps = 0.000001
    nerr = 0
    if (me==1) then
       if (abs(v)-abs(vv) > eps) then
          nerr = 1
       end if
    else
       if (abs(v)-abs((1.23d4, 5.67d8)) > eps) then
          nerr = 1
       end if
    end if

    if (nerr==0) then 
       print '("result[",i0,"] OK")', me
    else
       print '("result[",i0,"] number of NGs: ",i0)', me, nerr
    end if
    !---------------------------- check and output end

  end program
