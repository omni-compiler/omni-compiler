  program test_a2_d2
    include "xmp_coarray.h"
    integer a2d2(10,8)[*]
    integer xmp_node_num
    integer nerr

    me = xmp_node_num()   ! == this_image()

    !---------------------------- switch on message
!!    call xmpf_coarray_msg(1)

    !---------------------------- initialization
    a2d2=-123
    sync all

    !---------------------------- exec
    if (me==1) then
       a2d2(1:3,1:8)[2] = (/ &
            11,21,31,12,22,32,13,23,33,&
            14,24,34,15,25,35,16,26,36,&
            17,27,37,18,28,38/)
    end if

    sync all

    !---------------------------- check and output start
    nerr = 0
    do i1=1,10
       do i2=1,8
!!          if (me==2.and.(i1==1.or.i1==3.or.i1==5.or.i1==7.or.i1==9)) then
          if (me==2.and.(i1==1.or.i1==2.or.i1==3)) then
             ival=(i1*10+i2)
          else
             ival=-123
          end if

          if (a2d2(i1,i2).ne.ival) then
             write(*,101) i1,i2,me,a2d2(i1,i2),ival
             nerr=nerr+1
          end if
       end do
    end do

    if (nerr==0) then 
       print '("[",i0,"] OK")', me
    else
       print '("[",i0,"] number of NGs: ",i0)', me, nerr
    end if
    !---------------------------- check and output end

101 format ("a2d2(",i0,",",i0,")[",i0,"]=",i0," should be ",i0)

  end program
