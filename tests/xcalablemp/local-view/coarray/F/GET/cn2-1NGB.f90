  program gettest_char_n
    include "xmp_coarray.h"
    character(len=5) c5(4,3)[*]
    character(len=9) c3(10,3)

    me=xmp_node_num()

    !---------------------------- switch on message
!!    call xmpf_coarray_msg(1)

    !---------------------------- initialization
    c3="MerryXmas"
    if (me==1) then
       c5(1,:)="abcde"
       c5(2,:)="fghij"
       c5(3,:)="Happy"
       c5(4,:)="Hoppy"
    else if (me==2) then
       c5(1,:)="01234"
       c5(2,:)="56789"
       c5(3,:)="01234"
       c5(4,:)="56789"
    else 
       c5(1,:)="abcde"
       c5(2,:)="NewYe"
       c5(3,:)="ukuki"
       c5(4,:)="wkwak"
    endif
    sync all

    !---------------------------- exec
    if (me==2) then
       !! bug182 c3(9,:)=c5(3,:)[1]//c5(2,:)[3]
       c3(9,1)=c5(3,1)[1]//c5(2,1)[3]
       c3(9,2)=c5(3,2)[1]//c5(2,2)[3]
       c3(9,3)=c5(3,3)[1]//c5(2,3)[3]
    end if
    sync all
    
    !---------------------------- check and output
    nerr = 0

    do j=1,3
    do i=1,10
       if (me==2.and.i==9) then
          if (c3(i,j).ne."HappyNewY") then
             write(*,101) i,j,me,c3(i,j),"HappyNewY"
             nerr=nerr+1
          end if
       else
          if (c3(i,j).ne."MerryXmas") then
             write(*,101) i,j,me,c3(i,j),"MerryXmas"
             nerr=nerr+1
          end if
       end if
    end do
    end do

101 format ("c3(",i0,",",i0,")[",i0,"]=/",a,"/ should be /",a,"/")

    if (nerr==0) then 
       print '("[",i0,"] OK")', me
    else
       print '("[",i0,"] number of NGs: ",i0)', me, nerr
    end if

  end program gettest_char_n
