  program putgettest
    include "xmp_coarray.h"
    integer*2 a2d2(10,8)[*]
    integer nerr

    me = this_image()
!!    if (me==1) call xmpf_coarray_msg(1)

    !---------------------------- switch on message
!!    call xmpf_coarray_msg(1)

    !---------------------------- initialization
    if (me==3) then
       do j=1,8
          do i=1,10
             a2d2(i,j)=-i*j
          enddo
       enddo
    else
       a2d2=0
    endif
    sync all

    !---------------------------- exec
    if (me==1) then
       !! -a2d2(1,2)[3] = 2, -a2d2(4,2)[3] = 8
       !! a2d2(3,1)[3] = -3
!!       a2d2(1:3,1:-a2d2(4,2)[3])[-a2d2(1,2)[3]] = &
       a2d2(1:3,1:int(-a2d2(4,2)[3]))[int(-a2d2(1,2)[3])] = &
            reshape((/ &
            11_2,21_2,31_2,12_2,22_2,32_2,13_2,23_2,33_2,&
            14_2,24_2,34_2,15_2,25_2,35_2,16_2,26_2,36_2,&
            17_2,27_2,37_2,18_2,28_2,38_2/), (/3,8/))
    end if

    sync all

    !---------------------------- check and output start
    nerr = 0
    do i2=1,8
       do i1=1,10
          if (me==2.and.(i1==1.or.i1==2.or.i1==3)) then
             ival=(i1*10+i2)
          else if (me==3) then
             ival=-i1*i2
          else
             ival=0
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
