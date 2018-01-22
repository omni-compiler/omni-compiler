  program a1d1
!!     include "xmp_coarray.h"
    integer*8 a(100)[*]
    integer b(100)
    integer nerr

    me = this_image()

    !---------------------------- switch on message
!!    call xmpf_coarray_msg(1)

    !---------------------------- initialization

    do i=1,100
       a(i) = i+me*100
    end do

    b = -99

    sync all

    !---------------------------- execution
    if (me==1) then
       b(11:44)=a(1:100:3)[2]
    else if (me==3) then
       b(11:44)=a(1:100:3)
    end if

    sync all

    !---------------------------- check and output start
    nerr = 0

    do i=1,100
       nval=i+me*100
       if (a(i).ne.nval) then
          write(*,101) i,me,a(i),nval
          nerr=nerr+1
       end if
    end do

    do i=1,100
       if (me==1.and.i>=11.and.i<=44) then
          nval=a(i*3-32)+100
       else if (me==3.and.i>=11.and.i<=44) then
          nval=a(i*3-32)
       else
          nval=-99
       end if
       if (b(i).ne.nval) then
          write(*,102) i,me,b(i),nval
          nerr=nerr+1
       end if
    end do

101 format ("a(",i0,")[",i0,"]=",i0," should be ",i0)
102 format ("b(",i0,")[",i0,"]=",i0," should be ",i0)

    if (nerr==0) then 
       print '("result[",i0,"] OK")', me
    else
       print '("result[",i0,"] number of NGs: ",i0)', me, nerr
    end if
    !---------------------------- check and output end

  end program
