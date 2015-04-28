  program test_a2_d0
    include "xmp_coarray.h"
    real a2(2,3)[*]
    integer xmp_node_num

    me = this_image()

    a2=7.77
    write(*,*) me,a2

    if (me==1) then
       a2(1,2)[2]=1.22
    else if (me==2) then
       a2(2,3)[3]=2.33
    end if

    sync all

    write(*,*) me,a2

  end program
