!--------------------------------------------------
! estimation of the sizes of structure components
!--------------------------------------------------

    type g0
       integer*8 a(10)
       real*8    r(10)
    end type g0

    type g1
       integer*8 a(10)                            !! + 80B (10elem)
       real*8    r(10)                            !! + 80B (10elem)
!!       type(g0),pointer :: p                    !! +  8B ( 1elem)
!!       type(g0),pointer :: p(:)                 !! + 48B ( 6elem)
!!       type(g0),pointer :: p(:,:)               !! + 72B ( 9elem)
!!       type(g0),pointer :: p(:,:,:)             !! + 96B (12elem)
!!       type(g0),pointer :: p(:,:,:,:)           !! +120B (15elem)
!!       type(g0),allocatable :: p                !! error
!!       type(g0),allocatable :: p(:)             !! + 48B ( 6elem)
!!       type(g0),allocatable :: p(:,:)           !! + 72B ( 9elem)
!!       type(g0),allocatable :: p(:,:,:)         !! + 96B (12elem)
!!       type(g0),allocatable :: p(:,:,:,:)       !! +120B (15elem)
!!       type(g0),allocatable :: p(:,:,:,:,:)     !! +144B (18elem)
       complex,allocatable :: p(:,:,:,:,:)     !! +144B (18elem)
!!       character(3),allocatable :: p(:,:,:,:,:)     !! +144B (18elem)
    end type g1

    type(g0) :: aa0
    type(g1) :: aa1

    write (*,*) "sizeof(aa0)=",sizeof(aa0)
    write (*,*) "sizeof(aa1)=",sizeof(aa1)
    write (*,*) "sizeof(aa1%a)=",sizeof(aa1%a)
    write (*,*) "sizeof(aa1%r)=",sizeof(aa1%r)
    write (*,*) "sizeof(aa1%p)=",sizeof(aa1%p)

    end
