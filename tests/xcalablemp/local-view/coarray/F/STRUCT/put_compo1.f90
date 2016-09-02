  program get1

    type g1
       integer*8 a(10)
       real*8    r(10)
    end type g1

    type(g1), save :: ns1, cs1[*]
!!    type(g1), save :: na1(3), ca1(3)[*]
    real*8 zz, eps, r(10)
    integer*8 nval
    real*8    dcal

    me = this_image()

    !---------------------------- initialization

    do i=1,10
       ns1%a(i) = i*me
       cs1%a(i) = i*me
       ns1%r(i) = 1.0d0 / real(me*i,8)
       cs1%r(i) = 1.0d0 / real(me*i,8)
    enddo

    sync all

    !---------------------------- execution
    if (me==3) then
       cs1[2]%r(9) = cs1%a(10) * cs1%r(2)   !! (10*3) * (1/(3*2)) = 5.0d0
    endif
    sync all

    !---------------------------- check and output start
    nerr = 0
    eps = 0.00001

    r = cs1%r
    do i=1,10
       if (i==9 .and. me==2 .and. abs(r(i)-5.0d0) > eps) then
          nerr=nerr+1
          write(*,103) me,"r(i)",5.0d0,r(i)
       else if (r(i) /= 1.0d0 / real(me*i,8)) then
          nerr=nerr+1
          write(*,103) me,"r(i)",5.0d0,r(i)
       end if
    end do

103 format ("[",i0,"] ",a," should be ",d12.8," but ",d12.8)
104 format ("[",i0,"] cs1%r(",i0,") should be ",d12.8," but ",d12.8)
105 format ("[",i0,"] cs1%a(",i0,") should be ",i0," but ",i0)

    if (nerr==0) then 
       print '("result[",i0,"] OK")', me
    else
       print '("result[",i0,"] number of NGs: ",i0)', me, nerr
    end if
    !---------------------------- check and output end

  end program
