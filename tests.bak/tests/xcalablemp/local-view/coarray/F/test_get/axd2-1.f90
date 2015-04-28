  program test_a1_d1
    include "xmp_coarray.h"

    integer*8 a(7,3:3)[2,*], b(10,3)
    integer nerr

    me = this_image()

    !---------------------------- switch on message
!!    if (me==2) call xmpf_coarray_msg(1)

    !---------------------------- initialization
    b = -1234

    do i=1,7
       a(i,3) = i*me
    end do

    sync all

    !---------------------------- execution
    if (me==1) then
       b(1:7,1:1)=a[2,1]
    end if

    sync all

    !---------------------------- check and output start
    nerr = 0
    do j=1,3
       do i=1,10
          if (me==1.and.j==1.and.(1<=i.and.i<=7)) then
             nval = i*2
          else
             nval = -1234
          end if

          if (nval.ne.b(i,j)) then
             write(*,101) i,j,me,b(i,j),nval
             nerr=nerr+1
          end if
       end do
    end do

    if (nerr==0) then 
       print '("result[",i0,"] OK")', me
    else
       print '("result[",i0,"] number of NGs: ",i0)', me, nerr
    end if
    !---------------------------- check and output end

101 format ("b(",i0,",",i0,")[",i0,"]=",i0," should be ",i0)

  end program
