  program implicit_type_conv
    include "xmp_coarray.h"
    real*8 a(5)
    real*4 b(5)[*]
    integer nerr

    me = this_image()

    !---------------------------- switch on message
!!    call xmpf_coarray_msg(1)

    !---------------------------- initialization
    a = 1.23456789
    b = 5.678
    sync all

    !---------------------------- execution
    if (me==2) then
       b[1]=a
    end if
    sync all

    !---------------------------- check and output start
    eps = 0.0001
    nerr = 0

    do i=1,5
       if (me==1) then
          val = 1.23456789
       else
          val = 5.678
       end if

       if (val-eps<b(i).and.b(i)<val+eps) then
          continue                                 ! OK
       else
          nerr=nerr+1
          write(*,101) i,me,b(i),val
       end if
    end do

    if (nerr==0) then 
       print '("result[",i0,"] OK")', me
    else
       print '("result[",i0,"] number of NGs: ",i0)', me, nerr
    end if

101 format ("b(",i0,")[",i0,"]=",f10.6," should be ",f10.6)

  end program
