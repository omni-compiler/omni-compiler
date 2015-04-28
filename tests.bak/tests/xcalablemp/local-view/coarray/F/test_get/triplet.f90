  program gettest_triplet
    include "xmp_coarray.h"

    integer*8 a(10)[2,*], b(10)
    integer nerr

    me = this_image()

    !---------------------------- switch on message
!!    if (me==2) call xmpf_coarray_msg(1)

    !---------------------------- initialization
    b = -1234

    do i=1,10
       a(i) = i*me
    end do

    sync all

    !---------------------------- execution
    if (me==1) then
       !! a(4)[2,1]=8
       b(3:a(4)[2,1]:2)=me*7
    end if

    sync all

    !---------------------------- check and output start
    nerr = 0

    do i=1,10
       if (me==1.and.(i==3.or.i==5.or.i==7)) then
          ival=7
       else
          ival=-1234
       endif
       if (b(i).ne.ival) then
          write(*,101) i,me,b(i),ival
          nerr=nerr+1
       end if
    end do

    if (nerr==0) then 
       print '("result[",i0,"] OK")', me
    else
       print '("result[",i0,"] number of NGs: ",i0)', me, nerr
    end if
    !---------------------------- check and output end

101 format ("b(",i0,")[",i0,"]=",i0," should be ",i0)

  end program
