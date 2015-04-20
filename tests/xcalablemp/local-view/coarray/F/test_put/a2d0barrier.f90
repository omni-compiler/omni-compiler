  program test_a2_d0
    include "xmp_coarray.h"
    !$xmp nodes p(*)
    real a2(2,3)[*]
    integer xmp_node_num

    me = xmp_node_num()   ! == this_image()

    a2=7.77
    write(*,*) me,a2

    if (me==1) then
       a2(1,2)[2]=1.22
    else if (me==2) then
       a2(2,3)[3]=2.33
    end if

!$xmp barrier
    sync all
!$xmp barrier
    write(*,*) me,a2

  end program
