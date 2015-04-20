  program gettest_cn0d0
    include "xmp_coarray.h"

    character(200) wbuf1,wbuf2[*],tmp

    me = xmp_node_num()   ! == this_image()

    !---------------------------- switch on message
!!    call xmpf_coarray_msg(1)

    !---------------------------- initialization
    wbuf1="War is over if you want it."
    wbuf2="Happy Xmas."
    sync all

    !---------------------------- exec
    if (me==1) tmp = wbuf2[2]
    sync all

    !---------------------------- check and output
    if (me==1) then
       if (wbuf2.ne.tmp) then
          write(*,*) "[1] tmp should be the same as my wbuf2."
          nerr=nerr+1
       end if
    end if

    if (nerr==0) then 
       print '("result[",i0,"] OK")', me
    else
       print '("result[",i0,"] number of NGs: ",i0)', me, nerr
    end if

  end program
