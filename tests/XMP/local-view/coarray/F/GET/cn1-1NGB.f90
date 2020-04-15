  program gettest_char_n
!!     include "xmp_coarray.h"
    character(len=5) c5(4)[*]
    character(len=9) c3(10)

    me=xmp_node_num()
    if (xmpf_coarray_uses_fjrdma()) then
       write(*,'(a)') "Using FJRDMA ... stop"
       stop
    endif

    !---------------------------- switch on message
!!    call xmpf_coarray_msg(1)

    !---------------------------- initialization
    c3="MerryXmas"
    if (me==1) then
       c5(1)="abcde"
       c5(2)="fghij"
       c5(3)="Happy"
       c5(4)="Hoppy"
    else if (me==2) then
       c5(1)="01234"
       c5(2)="56789"
       c5(3)="01234"
       c5(4)="56789"
    else 
       c5(1)="abcde"
       c5(2)="NewYe"
       c5(3)="ukuki"
       c5(4)="wkwak"
    endif
    sync all

    !---------------------------- exec
    if (me==2) then
       c3(9)=c5(3)[1]//c5(2)[3]
    end if
    sync all
    
    !---------------------------- check and output
    nerr = 0

    do i=1,10
       if (me==2.and.i==9) then
          if (c3(i).ne."HappyNewY") then
             write(*,101) i,me,c3(i),"HappyNewY"
             nerr=nerr+1
          end if
       else
          if (c3(i).ne."MerryXmas") then
             write(*,101) i,me,c3(i),"MerryXmas"
             nerr=nerr+1
          end if
       end if
    end do


101 format ("c3(",i0,")[",i0,"]=/",a,"/ should be /",a,"/")

    if (nerr==0) then 
       print '("[",i0,"] OK")', me
    else
       print '("[",i0,"] number of NGs: ",i0)', me, nerr
       call exit(1)
    end if

  end program gettest_char_n
