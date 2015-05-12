  program allo_loop
    include "xmp_coarray.h"
    real, allocatable :: abc(:,:)[:]

    do i=1,100000
       allocate(abc(1000,1000)[*])
       deallocate(abc)
    end do

    nerr = 0
    na = xmpf_coarray_allocated_bytes()
    ng = xmpf_coarray_garbage_bytes()
    if (na /= 0 .or. ng /= 0) then 
       nerr = 1
       write(*,100) this_image(), na, ng
    endif
100 format("[",i0,"] remains allocated ",i0," and gabage ",i0," bytes")

    call final_msg(na)

  end program allo_loop


  subroutine final_msg(nerr)
    include 'xmp_coarray.h'
    if (nerr==0) then 
       print '("[",i0,"] OK")', this_image()
    else
       print '("[",i0,"] number of NGs: ",i0)', this_image(), nerr
    end if
    return
  end subroutine final_msg
