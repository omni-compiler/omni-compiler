  program get1
    implicit none
    integer i,j,k,nerr,me,nval
    real eps,dval

    type g0
       integer*8 a(10)
       real*8    r(10)
    end type g0

    type g1
       integer*8 a(10)
       real*8    r(10)
       type(g0) :: s(2)
       type(g0),pointer :: p(:)
    end type g1

    real, save :: cr1[*]
    type(g1), save :: ns1, cs1[*]
    type(g1), save :: na1(3), ca1(3)[*]
    real*8 zz, eps, zz10(10)
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
    if (me==2) then
       zz = cs1[3]%a(10) * cs1[3]%r(2)   !! (10*3) * (1/(3*2)) = 5.0d0
    endif
    if (me==1) then
       do i=1,10
          cs1%a(i) = cs1%a(i) + cs1[2]%a(i) + cs1[3]%a(i)       !! i*(1+2+3)
          cs1%r(i) = 1/ns1%r(i) + 1/cs1[2]%r(i) + 1/cs1[3]%r(i) !! (1+2+3)*i
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

    do i=1,10
       if (me==1) then
          nval = i*6
          dval = real(6*i,8)
       else
          nval = i*me
          dval = 1.0d0 / real(me*i,8)
       endif

       if (cs1%a(i)-nval /= 0) then
          nerr=nerr+1
          write(*,105) me,i,nval,cs1%a(i)
       end if
       if (abs(cs1%r(i)-dval) > eps) then
          nerr=nerr+1
          write(*,104) me,i,dval,cs1%r(i)
       end if
    end do

103 format ("[",i0,"] ",a," should be ",d12.4," but ",d12.4)
105 format ("[",i0,"] cs1%a(",i0,") should be ",i0," but ",i0)
104 format ("[",i0,"] cs1%r(",i0,") should be ",d12.4," but ",d12.4)

    if (nerr==0) then 
       print '("result[",i0,"] OK")', me
    else
       print '("result[",i0,"] number of NGs: ",i0)', me, nerr
    end if
    !---------------------------- check and output end

  end program
