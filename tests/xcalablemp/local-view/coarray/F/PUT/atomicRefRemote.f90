  program main

    logical la(3,2)[*], laok, lb

    la=.false.
    lb=.false.
    me=this_image()
    sync all

    nerr=0

    !!------------------- TRY LOCAL-REMOTE
    if (me==3) then
       do while (.not.lb)
          call atomic_ref(lb, la(1,2)[2])
       enddo
       sync memory
    endif

    if (me==2) then
       sync memory
       call atomic_define(la(1,2), .true.)
!!       call atomic_define(la(1,2)[2], .true.)
    endif
    
    !!------------------- CHECK
    if (me==3) then
       if (.not.lb) then
          nerr=nerr+1
          write(*,101) me, "lb", .true., lb
       endif
    else
       if (lb) then
          nerr=nerr+1
          write(*,101) me, "lb", .false., lb
       endif
    endif

    do j=1,2
       do i=1,3
          laok=.false.
          if (i==1 .and. j==2 .and. me==2) laok=.true.

          if (la(i,j).neqv.laok) then
             nerr=nerr+1
             write(*,102) me, "la", i, j, laok, la(i,j)
          endif
       enddo
    enddo

    !!------------------- SUMMARY OUTPUT
    if (nerr==0) then 
       print '("[",i0,"] OK")', me
    else
       print '("[",i0,"] number of NGs: ",i0)', me, nerr
    end if


100 format("[",i0,"] ",a," should be ",i0," but ",i0,".")
101 format("[",i0,"] ",a," should be ",l2," but ",l2,".")
102 format("[",i0,"] ",a,"(",i0,",",i0,") should be ",l2," but ",l2,".")

  end program main

