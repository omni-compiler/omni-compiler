  program times2
    include "xmp_coarray.h"
    real, allocatable :: a(:,:)[:]
    real, allocatable :: b(:,:)[:]
    real, allocatable :: c(:,:)[:]
    real, allocatable :: d(:,:)[:]
    me = this_image()

    do i=1,10000

       allocate(a(1000,569)[*],b(1000,800)[*])
       deallocate(a)
       allocate(c(1300,1000)[*],d(1001,1001)[*])
       deallocate(d,b)
       deallocate(c)

    end do

    nerr = 0
    na = xmpf_coarray_allocated_bytes()
    ng = xmpf_coarray_garbage_bytes()
    if (na /= 0 .or. ng /= 0) then 
       nerr = 1
       write(*,100) this_image(), na, ng
    endif

    call final_msg(na)

100 format("[",i0,"] remains allocated ",i0," and gabage ",i0," bytes")

  end program times2


  subroutine final_msg(nerr)
    include 'xmp_coarray.h'
    if (nerr==0) then 
       print '("[",i0,"] OK")', this_image()
    else
       print '("[",i0,"] number of NGs: ",i0)', this_image(), nerr
    end if
    return
  end subroutine final_msg

