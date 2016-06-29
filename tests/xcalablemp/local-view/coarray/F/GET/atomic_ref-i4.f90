  program a1d1
!!     include "xmp_coarray.h"
    integer*4 a(100)[*]
    integer b(100)
    integer nerr

    me = this_image()

    !---------------------------- initialization

    do i=1,100
       a(i) = i+me*100
    end do

    b = -99

    sync all

    !---------------------------- execution
    if (me==1) then
       call atomic_ref(b(11),a(3)[2])
    else if (me==3) then
       call atomic_ref(b(11),a(3))
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
       if (me==1.and.i==11) then
          nval=203
       else if (me==3.and.i==11) then
          nval=303
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
