  program test_a3_d3
    include "xmp_coarray.h"
    integer a3(10,1,8)[*]
    integer xmp_node_num
    integer nerr

    me = xmp_node_num()   ! == this_image()

    !---------------------------- switch on message
!!    call xmpf_coarray_msg(1)

    !---------------------------- initialization
    if (me==1) then
       do i1=1,10
          do i2=1,1
             do i3=1,8
                ival=(i1*10+i2)*10+i3
                a3(i1,i2,i3)=ival
             end do
          end do
       end do
    else
       a3=-123
    endif
    sync all

    !---------------------------- exec
    if (me==1) then
       a3(1:10:2,:,1:8)[2]=a3(1:10:2,:,1:8)
!!       a3(1:10,:,1:8)[2]=a3(1:10,:,1:8)
    end if

    sync all

    !---------------------------- check and output start
    nerr = 0
    if (me==2) then
       do i1=1,10,2
!!       do i1=1,10
          do i2=1,1
             do i3=1,8
                ival=(i1*10+i2)*10+i3
                jval=a3(i1,i2,i3)
                if (jval.ne.ival) then
                   write(*,101) i1,i2,i3,me,jval,ival
                   nerr=nerr+1
                end if
                a3(i1,i2,i3)=-123
             end do
          end do
       end do

       ival=-123
       do i1=1,10
          do i2=1,1
             do i3=1,8
                if (a3(i1,i2,i3).ne.ival) then
                   write(*,101) i1,i2,i3,me,a3(i1,i2,i3),ival
                   nerr=nerr+1
                end if
             end do
          end do
       end do
    end if

    if (nerr==0) then 
       print '("[",i0,"] OK")', me
    else
       print '("[",i0,"] number of NGs: ",i0)', me, nerr
    end if
    !---------------------------- check and output end

101 format ("a3(",i0,",",i0,",",i0,")[",i0,"]=",i0," should be ",i0)

  end program
