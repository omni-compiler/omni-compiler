  program main
    integer a(10)[*],b(10),c(10)
    integer,parameter:: nok(10)=(/1,2,10,20,30,40,50,60,0,0/)

    nerr=0
    me=this_image()

    do i=1,10
       a(i)=0
       b(i)=i
       c(i)=i*10
    enddo

    syncall

    if (me==1) then
       a(1:5)[2]=b(1:5)
       a(3:8)[2]=c(1:6)
    endif

    syncall

    if (me==2) then
!!!       write(*,*) a
       do i=1,10
          if (a(i).ne.nok(i)) nerr=nerr+1
       enddo
       if (nerr>0) write(*,100) me,nok, a
    endif

!---- OUTPUT
    if (nerr==0) then
       write(*,101) me
    else
       write(*,102) me, nerr
    endif

100 format("[",i0,"] a(:) should be ",10(i0,",")," but ",10(i0,","))
101 format("[",i0,"] OK")
102 format("[",i0,"] NG  nerr=",i0)

  end program main

