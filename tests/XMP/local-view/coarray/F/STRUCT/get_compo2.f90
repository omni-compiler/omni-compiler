  program get1

    type g1
       integer*8 a(10)
       real*8    r(10)
    end type g1

!!    type(g1), save :: ns1, cs1[*]
    type(g1), save :: na1(3), ca1(3)[*]
    real*8 zz, eps
    integer*8 nval
    real*8    dcal

    me = this_image()

    !---------------------------- initialization

    do i=1,10
       na1%a(i) = i*me
       ca1%a(i) = i*me
       na1%r(i) = 1.0d0 / real(me*i,8)
       ca1%r(i) = 1.0d0 / real(me*i,8)
    enddo

    sync all

    !---------------------------- execution
    if (me==2) then
       zz = ca1(1)[3]%a(10) * ca1(1)[3]%r(2)   !! (10*3) * (1/(3*2)) = 5.0d0
    endif
    if (me==1) then
       do i=1,10
          ca1(2)%a(i) = ca1(2)%a(i) + ca1(2)[2]%a(i) + ca1(2)[3]%a(i)    !! i*(1+2+3)
          ca1(2)%r(i) = 1/na1%r(i) + 1/ca1(2)[2]%r(i) + 1/ca1(2)[3]%r(i) !! (1+2+3)*i
       enddo
    end if
    sync all

    !---------------------------- check and output start
    nerr = 0
    eps = 0.00001

    if (me==2 .and. abs(zz-5.0d0) > eps) then
       nerr=nerr+1
       write(*,103) me,"zz",5.0d0,zz
    endif

    do i=1,100
       if (me==1) then
          nval = i*6
          dcal = real(6*i,8)
       else
          nval = i*me
          dval = 1.0d0 / real(me*i,8)
       endif

       if (ca1(2)%a(i)-nval /= 0) then
          nerr=nerr+1
          write(*,105) me,i,nval,ca1(2)%a(i)
       end if
       if (abs(ca1(2)%r(i)-dval) > eps) then
          nerr=nerr+1
          write(*,104) me,i,dval,ca1(2)%r(i)
       end if
    end do

103 format ("[",i0,"] ",a," should be ",d12.8," but ",d12.8)
104 format ("[",i0,"] ca1(2)%r(",i0,") should be ",d12.8," but ",d12.8)
105 format ("[",i0,"] ca1(2)%a(",i0,") should be ",i0," but ",i0)

    if (nerr==0) then 
       print '("result[",i0,"] OK")', me
    else
       print '("result[",i0,"] number of NGs: ",i0)', me, nerr
    end if
    !---------------------------- check and output end

  end program
