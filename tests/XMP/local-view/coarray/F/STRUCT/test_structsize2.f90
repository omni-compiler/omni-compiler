!--------------------------------------------------
! estimation of the sizes of structure components
!--------------------------------------------------

    type g1
!!       integer*8 a(10)             !! + 80B              = 80B
       integer*4 a(20)             !! + 80B              = 80B
       real*8    r(10)             !! + 80B              =160B
!!       real*4    r(20)             !! + 80B              =160B
       character(3) :: c(3)        !! +  9B  + round-up  =172B
       real*4    q(2)              !! +  8B  + round-up  =184B
    end type g1

    type(g1) :: aa1, aa2(2)

    write (*,*) "sizeof(aa1)=",sizeof(aa1)
    write (*,*) "sizeof(aa1%a)=",sizeof(aa1%a)
    write (*,*) "sizeof(aa1%r)=",sizeof(aa1%r)
    write (*,*) "sizeof(aa1%c)=",sizeof(aa1%c)
    write (*,*) "sizeof(aa1%q)=",sizeof(aa1%q)
    write (*,*) "sizeof(aa2)=",sizeof(aa2)

    end
