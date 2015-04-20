  program test_a1_d1
    include "xmp_coarray.h"

    integer*8 a(10)[2,*], b(10)
    integer nerr

    me = this_image()

    !---------------------------- switch on message
!!    if (me==2) call xmpf_coarray_msg(1)

    !---------------------------- initialization
    b = -1234

    do i=1,10
       a(i) = i
    end do

    sync all

    !---------------------------- execution
    if (me==1) then
       b(1:7)=a(1:7)[2,1]
    end if

    sync all

    !---------------------------- check and output start
    nerr = 0
    if (me==1) then
       do i=1,7
          ival=a(i)
          jval=b(i)
          if (jval.ne.ival) then
             write(*,101) i,me,jval,ival
             nerr=nerr+1
          end if
       end do

       do i=8,10
          if (b(i).ne.-1234) then
             write(*,101) i,me,b(i),-1234
             nerr=nerr+1
          end if
       end do

    else
       do i=1,10
          if (b(i).ne.-1234) then
             write(*,101) i,me,b(i),-1234
             nerr=nerr+1
          end if
       end do

    end if

!!    write(*,102) me,a
!!    write(*,103) me,b

    if (nerr==0) then 
       print '("result[",i0,"] OK")', me
    else
       print '("result[",i0,"] number of NGs: ",i0)', me, nerr
    end if
    !---------------------------- check and output end

101 format ("b(",i0,")[",i0,"]=",i0," should be ",i0)
102 format ("me=",i0," a=",10(i0,","))
103 format ("me=",i0," b=",10(i0,","))

  end program
