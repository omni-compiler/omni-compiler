  program pipo3
    include "xmp_lib.h"
    real, allocatable :: abc(:,:)[:]
    real, allocatable :: def(:,:)[:]
    real, allocatable :: ghi(:,:)[:]
    me = this_image()


    allocate(abc(2000,1000)[*])
    write(*,100) 8, xmpf_coarray_allocated(),  xmpf_coarray_suspended()

    allocate(def(2000,1000)[*])
    write(*,100) 16, xmpf_coarray_allocated(),  xmpf_coarray_suspended()

    allocate(ghi(2000,1000)[*])
    write(*,100) 24, xmpf_coarray_allocated(),  xmpf_coarray_suspended()

    deallocate(def)
    write(*,100) 24, xmpf_coarray_allocated(),  xmpf_coarray_suspended()

    deallocate(ghi)
    write(*,100) 8, xmpf_coarray_allocated(),  xmpf_coarray_suspended()

    allocate(ghi(1000,1000)[*])
    write(*,100) 12, xmpf_coarray_allocated(),  xmpf_coarray_suspended()

    deallocate(ghi)
    write(*,100) 8, xmpf_coarray_allocated(),  xmpf_coarray_suspended()

    deallocate(abc)
    write(*,100) 0, xmpf_coarray_allocated(),  xmpf_coarray_suspended()


100 format("i=",i0,": allocated ",i0," bytes, suspended ",i0," bytes")

  end program 
