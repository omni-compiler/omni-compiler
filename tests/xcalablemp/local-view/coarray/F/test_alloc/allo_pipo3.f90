  program pipo3
    include "xmp_coarray.h"
    real, allocatable :: abc(:,:)[:]
    real*8, allocatable :: def(:,:)[:]
    complex*16, allocatable :: ghi(:,:)[:]

    nerr=0

    allocate(abc(200,1000)[*])
    call check(200*1000*4, 0, "abc 1", nerr)

    allocate(def(200,1000)[*])
    call check(200*1000*4 + 200*1000*8, 0, "def 1", nerr)

    allocate(ghi(200,1000)[*])
    call check(200*1000*4 + 200*1000*8 + 200*1000*16, 0, "ghi 1", nerr)

    deallocate(def)
    call check(200*1000*4 + 200*1000*8 + 200*1000*16, 200*1000*8, "def de1", nerr)

    deallocate(ghi)
    call check(200*1000*4, 0, "ghi de1", nerr)

    allocate(ghi(1000,1000)[*])
    call check(200*1000*4 + 1000*1000*16, 0, "ghi 2", nerr)

    deallocate(ghi)
    call check(200*1000*4, 0, "ghi de2", nerr)

    deallocate(abc)
    call check(0, 0, "abc free-all", nerr)

    call final_msg(nerr)

  end program 

  subroutine check(alloc, garbage, msg, nerr)
    implicit none
    include "xmp_coarray.h"
    integer alloc, garbage, nerr
    character(*) msg
    integer nerr1, alloc1, garbage1, me

    me = this_image()
    nerr1 = 0

    alloc1 = xmpf_coarray_allocated_bytes()
    garbage1 = xmpf_coarray_garbage_bytes()

    if (alloc1 /= alloc) then
       nerr1 = 1
       write(*,101) me, msg, alloc1, alloc
    endif

    if (garbage1 /= garbage) then
       nerr1 = 1
       write(*,102) me, msg, garbage1, garbage
    endif

    nerr = nerr + nerr1
    return

101 format("[",i0,"] '",a,"' allocated should be ",i0," but ",i0)
102 format("[",i0,"] '",a,"' garbage should be ",i0," but ",i0)

  end subroutine check

  subroutine final_msg(nerr)
    include 'xmp_coarray.h'
    if (nerr==0) then 
       print '("[",i0,"] OK")', this_image()
    else
       print '("[",i0,"] number of NGs: ",i0)', this_image(), nerr
    end if
    return
  end subroutine final_msg

