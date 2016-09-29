!! level-1 release
!! restriction:
!!  1. A pointer component of a derived-type coarray is not allowed.
!!  2. An allocatable component of a derived-type coarray cannnot be 
!!     referrenced as a coindexed object.
!!  3. A derived-type coarray cannot be defined as allocatable.

  program main

    type ss1
       integer   :: c0(2,3)
       integer   :: d
    end type ss1

    type ss2
       integer   :: b0(2,3)
       type(ss1) :: b1
       type(ss1) :: b2(10)
    end type ss2

    type(ss2) :: a1[*]
    type(ss2) :: a2(10)[*]

!!    integer x1(3), x2(3), x3(3), x4(3)

    me=this_image()
    nerr=0

    !!------------------ init
    do j=1,3; do i=1,2
       a2(1)%b1%c0(i,j) = 100*me + (i+(j-1)*2)  !! [101,102;103,104;105,106] if me==1
       do k=1,10
          a2(2)%b2(k)%c0(i,j) =  100*me + 10*k + (i+(j-1)*2)
       enddo
    enddo; enddo
    sync all

    !!------------------ action
    if (me==1) then
       a1%b0(1,:) = a2(1)[3]%b1%c0(1,:)     !! [301,303,305]
       a1%b0(2,:) = a2(1)[2]%b1%c0(1,:)     !! [201,203,205]
       a1%b2(1:3)%d = a2(1)[1]%b1%c0(2,:)     !! [102,104,106]
       a1%b2(4:6)%d = a2(1)[2]%b1%c0(2,:)     !! [202,204,206]
!! issue#37 bug in F_Front 
!!       a1%b2(2:4)%c0(1,1) = a2(1)[1]%b1%c0(2,:)     !! [102,104,106]
!!       a1%b2(5:7)%c0(1,1) = a2(1)[2]%b1%c0(2,:)     !! [202,204,206]
!!       x1 = a2(2)[3]%b2(1:3)%c0(1,2)        !! [313,323,333]
!!       x2 = a2(2)[3]%b2(1:3:2)%c0(1,1)      !! [311,331,351]
!!       x3 = a2(2)[3]%b2(7:3:-2)%c0(2,2)      !! [374,354,334]
    endif
    sync all

    !!------------------ check
    if (me==1) then
       if (a1%b0(1,1).ne.301) nerr=nerr+1
       if (a1%b0(1,2).ne.303) nerr=nerr+1
       if (a1%b0(1,3).ne.305) nerr=nerr+1
       if (a1%b0(2,1).ne.201) nerr=nerr+1
       if (a1%b0(2,2).ne.203) nerr=nerr+1
       if (a1%b0(2,3).ne.205) nerr=nerr+1

       if (a1%b2(1)%d.ne.102) then; nerr=nerr+1; print *,"a1%b2(1)%d=",a1%b2(1)%d; endif
       if (a1%b2(2)%d.ne.104) then; nerr=nerr+1; print *,"a1%b2(2)%d=",a1%b2(2)%d; endif
       if (a1%b2(3)%d.ne.106) then; nerr=nerr+1; print *,"a1%b2(3)%d=",a1%b2(3)%d; endif
       if (a1%b2(4)%d.ne.202) then; nerr=nerr+1; print *,"a1%b2(4)%d=",a1%b2(4)%d; endif
       if (a1%b2(5)%d.ne.204) then; nerr=nerr+1; print *,"a1%b2(5)%d=",a1%b2(5)%d; endif
       if (a1%b2(6)%d.ne.206) then; nerr=nerr+1; print *,"a1%b2(6)%d=",a1%b2(6)%d; endif

!!       if (a1%b2(2)%c0(1,1).ne.102) nerr=nerr+1
!!       if (a1%b2(3)%c0(1,1).ne.104) nerr=nerr+1
!!       if (a1%b2(4)%c0(1,1).ne.106) nerr=nerr+1
!!       if (a1%b2(5)%c0(1,1).ne.202) nerr=nerr+1
!!       if (a1%b2(6)%c0(1,1).ne.204) nerr=nerr+1
!!       if (a1%b2(7)%c0(1,1).ne.206) nerr=nerr+1
!!
!!       if (x1(1).ne.313) nerr=nerr+1
!!       if (x1(2).ne.323) nerr=nerr+1
!!       if (x1(3).ne.333) nerr=nerr+1
!!
!!       if (x2(1).ne.311) nerr=nerr+1
!!       if (x2(2).ne.331) nerr=nerr+1
!!       if (x2(3).ne.351) nerr=nerr+1
!!
!!       if (x3(1).ne.374) nerr=nerr+1
!!       if (x3(2).ne.354) nerr=nerr+1
!!       if (x3(3).ne.334) nerr=nerr+1
    endif

    !!------------------ summary
    if (nerr==0) then 
       print '("[",i0,"] OK")', me
    else
       print '("[",i0,"] number of NGs: ",i0)', me, nerr
    end if

  end program main
