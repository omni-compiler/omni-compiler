  program cn1d1_write
    include "xmp_coarray.h"
    integer*2 a(100)[*], a_org(100)
    integer xmp_node_num
    integer nerr
    character(200) wbuf1, wbuf2[*], tmpbuf

    me = this_image()

    !---------------------------- switch on message
!!    call xmpf_coarray_msg(1)

    !---------------------------- initialization

    do i=1,100
       a_org(i) = i+me*100
       a(i) = a_org(i)
    end do

    sync all

    !---------------------------- execution
    if (me==1) then
!!       write(*,*) a(15:100:7)[2]
       write(wbuf1,*) a(15:100:7)[2]
    end if
    if (me==2) then
!!       write(*,*) a(15:100:7)
       write(wbuf2,*) a(15:100:7)
    end if

    sync all

    !---------------------------- check and output start

    nerr=0
    if (me==1) then
       tmpbuf = wbuf2[2]
       if (wbuf1.ne.tmpbuf) then
          write(*,*) "[1] wbuf1 and wbuf2[2] should be the same."
          nerr=nerr+1
       end if
    end if

    if (nerr==0) then 
       print '("result[",i0,"] OK")', me
    else
       print '("result[",i0,"] number of NGs: ",i0)', me, nerr
    end if
    !---------------------------- check and output end

  end program
