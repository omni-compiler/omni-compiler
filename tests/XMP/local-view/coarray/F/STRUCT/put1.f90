  program get1
    integer,parameter:: z=2

    type g1
       integer*2 ii(7)
       real*8    rr(8)
    end type g1

    type(g1), save :: ns1, cs1[*]
    type(g1), save :: na1(z+1:z+3), ca1(z+1:z+3)[*]

    type g2
       integer*2 ii
       real*8    rr
    end type g2

    type(g2) :: sok, aok

    !!---------------------- init
    me = this_image()

    ns1%ii = me*10
    ns1%rr = me*100.0
    do i=lbound(na1,1),ubound(na1,1)
       na1(i)%ii = me
       na1(i)%rr = me*0.1
    enddo

    cs1%ii = 0
    cs1%rr = 0.0
    do i=lbound(ca1,1),ubound(ca1,1)
       ca1(i)%ii = 0
       ca1(i)%rr = 0.0
    enddo

    sync all

    !!---------------------- action
    if (me==1) then
       cs1[2] = ns1
       ca1[3] = na1
    endif
    sync all

    !!---------------------- check
    nerr=0
    sok%ii = 0
    sok%rr = 0.0
    aok%ii = 0
    aok%rr = 0.0
    if (me==2) then
       sok%ii = 1*10
       sok%rr = 1*100.0
    else if (me==3) then
       aok%ii = 1
       aok%rr = 1*0.1
    endif

    do i=1,7
       if (cs1%ii(i) .ne. sok%ii) then
          nerr=nerr+1
          write(*,101) me, i, sok%ii, cs1%ii(i)
       endif
       if (cs1%rr(i) .ne. sok%rr) then
          nerr=nerr+1
          write(*,102) me, i, sok%rr, cs1%rr(i)
       endif
    enddo
 
    do j=lbound(ca1,1),ubound(ca1,1)
       do i=1,8
          if (ca1(j)%ii(i) .ne. sok%ii) then
             nerr=nerr+1
             write(*,103) me, j, i, sok%ii, cs1%ii(i)
          endif
          if (ca1(j)%rr(i) .ne. sok%rr) then
             nerr=nerr+1
             write(*,104) me, j, i, sok%rr, cs1%rr(i)
          endif
       enddo
    enddo

101 format ("[",i0,"] cs1%ii(",i0,") should be ",i0," but ",i0)
102 format ("[",i0,"] cs1%rr(",i0,") should be ",d12.8," but ",d12.8)
103 format ("[",i0,"] ca1(",i0,")%ii(",i0,") should be ",i0," but ",i0)
104 format ("[",i0,"] ca1(",i0,")%rr(",i0,") should be ",d12.8," but ",d12.8)


    if (nerr==0) then 
       print '("result[",i0,"] OK")', me
    else
       print '("result[",i0,"] number of NGs: ",i0)', me, nerr
    end if

  end program
