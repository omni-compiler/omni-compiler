  program this_image2
    include "xmp_coarray.h"

!$xmp nodes n1(2,4)
!$xmp nodes n2(2,2)=n1(:,3:4)
!$xmp nodes n3(2,2)=n1(2,:)

    real a(10,20)[2,*]
    me = this_image()
    write(*,*) "this_image(a)=",this_image(a)

!$xmp task on n2
    call two_way(a,1,me)
!$xmp end task
!$xmp task on n3
    call two_way(a,2,me)
!$xmp end task

  end program this_image2

  subroutine two_way(a,count,me)
    include 'xmp_coarray.h'

    integer count,me
    real a(10,20)[2,*],b(10,20)[2,*]

    write(*,*) "count,me,this_image(a)=",count,me,this_image(a)
    write(*,*) "count,me,this_image(b)=",count,me,this_image(b)

  end subroutine two_way

