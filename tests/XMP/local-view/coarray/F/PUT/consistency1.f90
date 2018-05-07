  program main
    integer a(1000)[*],b(1000),c(3),d(3)

    nerr=0
    me = this_image()

!---- TRY#1 with sync memory

    do i=1,1000
       a(i)=-i
       b(i)=i
    enddo
    c=0

    sync all

    if (me==1) then
       a(:)[2]=b(:)
       sync memory
       c(1)=a(1000)[2]
       c(2)=a(999)[2]
       c(3)=a(333)[2]
!!!       write(*,*) c
       nok=1000
       if (c(1).ne.nok) then
          nerr=nerr+1
          write(*,100) me,"c(1)",nok,c(1)
       end if
       nok=999
       if (c(2).ne.nok) then
          nerr=nerr+1
          write(*,100) me,"c(2)",nok,c(2)
       end if
       nok=333
       if (c(3).ne.nok) then
          nerr=nerr+1
          write(*,100) me,"c(3)",nok,c(3)
       end if
    endif


!---- TRY#2 without sync memory
    sync all

    do i=1,1000
       a(i)=-i
       b(i)=i
    enddo
    d=0

    sync all

    if (me==1) then
       a(:)[2]=b(:)
       d(1)=a(1000)[2]
       d(2)=a(999)[2]
       d(3)=a(333)[2]
!!!       write(*,*) d
       nok=1000
       if (d(1).ne.nok) then
          nerr=nerr+1
          write(*,100) me,"d(1)",nok,d(1)
       end if
       nok=999
       if (d(2).ne.nok) then
          nerr=nerr+1
          write(*,100) me,"d(2)",nok,d(2)
       end if
       nok=333
       if (d(3).ne.nok) then
          nerr=nerr+1
          write(*,100) me,"d(3)",nok,d(3)
       end if
    endif

    sync all


!---- OUTPUT
    if (nerr==0) then
       write(*,101) me
    else
       write(*,102) me, nerr
    endif

100 format("[",i0,"] ",a," should be ",i0," but ",i0)
101 format("[",i0,"] OK")
102 format("[",i0,"] NG  nerr=",i0)

  end program main

