!! level-1 release
!! restriction:
!!  1. A pointer component of a derived-type coarray is not allowed.
!!  2. An allocatable component of a derived-type coarray cannnot be 
!!     referrenced as a coindexed object.
!!  3. A derived-type coarray cannot be defined as allocatable.

  program main

    type z
       integer              :: n(2,3)
       integer,allocatable  :: na(:,:)
!!       integer,pointer      :: np(:)     !! restriction 1
    end type z

    integer              :: u[*], v(3)[*]
    type(z)              :: x[*], y(3)[*]

    integer, allocatable :: ua[:], va(:)[:]
!!    type(z), allocatable :: xa[:], ya(:)[:]   !! restriction 3
!!  type(z), pointer     :: xp[:], yp(:)[:]   not allowed in [J.Reid]

    me=this_image()
    nerr=0

    !!------------------ access in local
    if (me==1) then
       allocate (y(2)%na(3,2))
       do j=1,2; do i=1,3
          y(2)%na(i,j) = i+(j-1)*3    !! [1,2,3;4,5,6]
!!!          write(*,*) "init", y(2)%na(i,j)
       enddo; enddo

       do j=1,3; do i=1,2
          x%n(i,j) = y(2)%na(j,i)    !! [1,4;2,5;3,6]
!!!          write(*,*) "local1", x%n(i,j)
       enddo; enddo
    endif
    sync all

    !!------------------ get and put
    if (me==2) then
       allocate (x%na(2,2))
       x%na(:,1) = x[1]%n(:,3)         !! [3,6]
!!!       write(*,*) "get1", x%na(:,1)
       x%na(:,2) = x[1]%n(:,2)         !! [2,5]
!!!       write(*,*) "get2", x%na(:,2)
       y(1)[3]%n(1,1:2) = x%na(1,:)    !! [3,2]
!!!       write(*,*) "put1", x%na(1,:)
       y(2)[3]%n(2,2:3) = x%na(2,:)    !! [6,5]
!!!       write(*,*) "put2", x%na(2,:)
    endif
    sync all

    !!------------------ check
    if (me==1) then
       if (y(2)%na(1,1).ne.1) nerr=nerr+1
       if (y(2)%na(2,1).ne.2) nerr=nerr+1
       if (y(2)%na(3,1).ne.3) nerr=nerr+1
       if (y(2)%na(1,2).ne.4) nerr=nerr+1
       if (y(2)%na(2,2).ne.5) nerr=nerr+1
       if (y(2)%na(3,2).ne.6) nerr=nerr+1
       if (x%n(1,1).ne.1) nerr=nerr+1
       if (x%n(2,1).ne.4) nerr=nerr+1
       if (x%n(1,2).ne.2) nerr=nerr+1
       if (x%n(2,2).ne.5) nerr=nerr+1
       if (x%n(1,3).ne.3) nerr=nerr+1
       if (x%n(2,3).ne.6) nerr=nerr+1
    else if (me==2) then
       if (x%na(1,1).ne.3) nerr=nerr+1
       if (x%na(2,1).ne.6) nerr=nerr+1
       if (x%na(1,2).ne.2) nerr=nerr+1
       if (x%na(2,2).ne.5) nerr=nerr+1
    else if (me==3) then
       if (y(1)%n(1,1).ne.3) nerr=nerr+1
       if (y(1)%n(1,2).ne.2) nerr=nerr+1
       if (y(2)%n(2,2).ne.6) nerr=nerr+1
       if (y(2)%n(2,3).ne.5) nerr=nerr+1
    endif


    !!------------------ summary
    if (nerr==0) then 
       print '("[",i0,"] OK")', me
    else
       print '("[",i0,"] number of NGs: ",i0)', me, nerr
    end if

  end program main
