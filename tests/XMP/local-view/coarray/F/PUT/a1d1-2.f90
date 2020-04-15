  program test_a1_d1
!!     include "xmp_coarray.h"
    integer*8 a(100), b(100)[*]
    integer nerr

    me = this_image()

    !---------------------------- switch on message
!!    call xmpf_coarray_msg(1)

    !---------------------------- initialization
    b = -1234
    a = -5678

    do i=1,100
       a(i) = i
    end do

    sync all

    !---------------------------- execution
    if (me==2) then
       b(1:100:3)[1]=a(1:34)
    end if

    sync all

    !---------------------------- check and output start
    nerr = 0
    if (me==1) then
       do i=1,34
          ival=a(i)
          jval=b(i*3-2)
          if (jval.ne.ival) then
             write(*,101) i*3-2,me,jval,ival
             nerr=nerr+1
          end if
          b(i*3-2) = -1234
       end do

       do i=1,100
          if (b(i).ne.-1234) then
             write(*,101) i,me,b(i),-1234
             nerr=nerr+1
          end if
       end do
    end if

    if (nerr==0) then 
       print '("result[",i0,"] OK")', me
    else
       print '("result[",i0,"] number of NGs: ",i0)', me, nerr
       call exit(1)
    end if
    !---------------------------- check and output end

101 format ("b(",i0,")[",i0,"]=",i0," should be ",i0)

  end program
