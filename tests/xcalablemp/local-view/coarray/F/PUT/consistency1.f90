  program main
    integer a(1000)[*],b(1000),c1,c2,c3

    do i=1,1000
       a(i)=-i
       b(i)=i
    enddo
    c1=0
    c2=0
    c3=0

    sync all

    if (this_image()==1) then
       a(:)[2]=b(:)
       sync memory
       c1=a(1000)[2]
       c2=a(999)[2]
       c3=a(333)[2]
       write(*,*) c1,c2,c3
    endif

    sync all

    do i=1,1000
       a(i)=-i
       b(i)=i
    enddo
    c1=0
    c2=0
    c3=0

    sync all

    if (this_image()==1) then
       a(:)[2]=b(:)
       c1=a(1000)[2]
       c2=a(999)[2]
       c3=a(333)[2]
       write(*,*) c1,c2,c3
    endif

  end program main

