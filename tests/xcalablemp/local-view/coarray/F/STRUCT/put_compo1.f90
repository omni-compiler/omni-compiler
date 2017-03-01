  program get1

    type ga
       integer n(3)
       real    r(3)
    end type ga
    type g1
       integer*8 a(10)
       real*8    r(10)
       type(ga) :: g0(2)
    end type g1

    real, save :: rrr1[*]
    type(g1), save :: ns1, cs1[*]
!!    type(g1), save :: na1(3), ca1(3)[*]
    real*8 zz, eps, r(10)

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
       cs1[2]%r(7) = 1.0d0       !! type match:    r8 = r8
       cs1[2]%r(8) = 1.0         !! type mismatch: r8 = r4
       cs1[2]%g0(1)%r(2) = 1.0   !! type match:    r4 = r4
       cs1[2]%g0(1)%r(2) = 1.0_8 !! type mismatch: r4 = r8
       zz = cs1[2]%r(9)
       zz = rrr1[2]
       n = cs1[2]%g0(1)%n(3)
       cs1[2]%g0(1)%n(3) = 1.0
       cs1[2]%r(9) = cs1%a(10) * cs1%r(2)   !! (10*3) * (1/(3*2)) = 5.0d0
    endif
    sync all

    !---------------------------- check and output start
    nerr = 0
    eps = 0.0000001

    r = cs1%r
    do i=1,10
       if (i==7 .and. me==2) then
          if (abs(r(i)-1.0d0) > eps) then
             nerr=nerr+1
             write(*,104) me,i,1.0d0,r(i)
          endif
       else if (i==8 .and. me==2) then
          if (abs(r(i)-1.0) > eps) then
             nerr=nerr+1
             write(*,104) me,i,real(1.0,8),r(i)
          endif
       else if (i==9 .and. me==2) then
          if (abs(r(i)-5.0d0) > eps) then
             nerr=nerr+1
             write(*,104) me,i,5.0d0,r(i)
          endif
       else
          if (abs(r(i)-1.0d0/real(me*i,8)) > eps) then
             nerr=nerr+1
             write(*,104) me,i,1.0d0/real(me*i,8),r(i)
          end if
       end if
    end do

104 format ("[",i0,"] cs1%r(",i0,") should be ",d12.5," but ",d12.5)

    if (nerr==0) then 
       print '("result[",i0,"] OK")', me
    else
       print '("result[",i0,"] number of NGs: ",i0)', me, nerr
    end if
    !---------------------------- check and output end

  end program
