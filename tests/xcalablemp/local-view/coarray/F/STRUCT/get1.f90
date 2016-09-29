  program get1

    type g1
       integer*8 a(3)
       real*8    r(3)
    end type g1

    type(g1), save :: s2, s1[*]
    real*8 zz, eps
    integer*8 nval
    real*8    dcal

    me = this_image()

    !---------------------------- initialization

    do i=1,3
       s1%a(i) = i*me
       s1%r(i) = 1.0/i/me
       s2%a(i) = 0
       s2%r(i) = 0.0
    enddo

    sync all

    !---------------------------- execution
    if (me==2) then
       s2 = s1[3]
    endif
    sync all

    !---------------------------- check and output start
    nerr = 0

    do i=1,3
       if (me==2) then
          if (s2%a(i).ne.i*3) then
             nerr=nerr+1
             write(*,101) me, "s2%a", i, i*3, s2%a(i)
          endif
          if (s2%r(i).ne.1.0/i/me) then
             nerr=nerr+1
             write(*,102) me, "s2%r", i, 1.0/i/3, s2%r(i)
          endif
       else
          if (s2%a(i).ne.i*3) then
             nerr=nerr+1
             write(*,101) me, "s2%a", i, 0, s2%a(i)
          endif
          if (s2%r(i).ne.1.0/i/me) then
             nerr=nerr+1
             write(*,102) me, "s2%r", i, 0.0, s2%r(i)
          endif
       end if
    end do

101 format ("[",i0,"] ",a,"(",i0,") should be ",i0," but ",i0)
102 format ("[",i0,"] ",a,"(",i0,") should be ",d12.8," but ",d12.8)

    if (nerr==0) then 
       print '("result[",i0,"] OK")', me
    else
       print '("result[",i0,"] number of NGs: ",i0)', me, nerr
    end if

  end program
