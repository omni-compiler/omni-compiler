  program test_a2_d2
    include "xmp_coarray.h"
    integer a2d2(10,8)[*]
    integer xmp_node_num
    integer nerr

    me = xmp_node_num()   ! == this_image()

    !---------------------------- switch on message
!!    call xmpf_coarray_msg(1)

    !---------------------------- initialization
    if (me==1) then
       do i2=1,8
          do i1=1,10
             ival=(i1*10+i2)
             a2d2(i1,i2)=ival
          end do
       end do
    else
       a2d2=-123
    endif
    sync all

    !---------------------------- exec
    if (me==1) then
       a2d2(1:10:2,1:8)[2]=a2d2(1:10:2,1:8)
    end if

    sync all

    !---------------------------- check and output start
    nerr = 0
    if (me==2) then
       do i2=1,8
          do i1=1,10,2
             ival=(i1*10+i2)
             jval=a2d2(i1,i2)
             if (jval.ne.ival) then
                write(*,101) i1,i2,me,jval,ival
                nerr=nerr+1
             end if
             a2d2(i1,i2)=-123
          end do
       end do

       ival=-123
       do i2=1,8
          do i1=1,10
             if (a2d2(i1,i2).ne.ival) then
                write(*,101) i1,i2,me,a2d2(i1,i2),ival
                nerr=nerr+1
             end if
          end do
       end do
    end if

    if (nerr==0) then 
       print '("[",i0,"] OK")', me
    else
       print '("[",i0,"] number of NGs: ",i0)', me, nerr
       stop 1
    end if
    !---------------------------- check and output end

101 format ("a2d2(",i0,",",i0,")[",i0,"]=",i0," should be ",i0)

  end program
