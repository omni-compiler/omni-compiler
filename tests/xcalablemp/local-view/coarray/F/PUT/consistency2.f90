  program main
    integer a(10)[*],b(10),c(10)

    do i=1,10
       a(i)=0
       b(i)=i
       c(i)=i*10
    enddo

    syncall

    if (this_image()==1) then
       a(1:5)[2]=b(1:5)
       a(3:8)[2]=c(1:6)
    endif

    syncall

    if (this_image()==2) then
       write(*,*) a
    endif

  end program main

