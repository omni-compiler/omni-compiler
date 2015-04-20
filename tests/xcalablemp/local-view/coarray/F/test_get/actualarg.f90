  integer function foo(n1,n2,n3)
    integer n1, n3
    integer*8 n2
    foo = (n1+n2)/2+n3
    return
  end function foo

  program gettest_actualarg
    include "xmp_coarray.h"

    integer*8 a(10)[2,*], b(10)
    integer xmp_node_num
    integer nerr

    me = xmp_node_num()   !! this_image()

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
       !! a(3)[1,2]=9, foo(2,9,1)=6
       !! a(5)[2,1]=10
       b(a(5)[2,1])=foo(2,a(3)[1,2],1)
    end if

    sync all

    !---------------------------- check and output start
    nerr = 0

    do i=1,10
       if (me==1.and.i==10) then
          ival=6
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
