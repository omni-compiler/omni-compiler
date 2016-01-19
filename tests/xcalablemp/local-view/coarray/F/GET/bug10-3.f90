  include "xmp_coarray.h"
  integer a(20)[*]

  me = xmp_node_num()
  do i=1,20
     a(i) = i*me
  enddo

  sync all
  call wait_seconds(10)
  sync all

  n1 = a(10)[1]
  n2 = a(10)[2]
  n3 = a(10)[3]

  sync all
  call wait_seconds(10)
  sync all
  write(*,*) "me,n1,n2,n3=",me,n1,n2,n3
  end


  subroutine wait_seconds(ns)
    integer ns
    real t1, t2, t3

    call cpu_time(t1)
    t3 = t1 + real(ns)
    do
       call cpu_time(t2)
       if (t2 >= t3) exit
    enddo
       
    return
  end subroutine wait_seconds
