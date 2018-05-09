  program main
    integer nsize, xsize, nerr, nwarn

    type z
       integer              :: n1(10)        !!    4*10 = 40
       integer,allocatable  :: na(:)         !!  24r+24 = 48
       integer,pointer      :: np(:)         !!  24r+24 = 48
    end type z                               !!  total   136 at most

    integer              :: u[*], v(10)[*]
    type(z)              :: x[*], y(10)[*]

    integer, allocatable :: ua[:], va(:)[:]
    type(z), allocatable :: xa[:], ya(:)[:]
!!  type(z), pointer     :: xp[:], yp(:)[:]   not allowed in [J.Reid]

    me=this_image()
    nerr=0
    nwarn=0

    !!------------------ before alloc
    nsize = 4+40+136+1360
    xsize = xmpf_coarray_allocated_bytes()
    if (nsize < xsize) then
       nerr=nerr+1
       write(*,105) me, nsize, xsize
    else if (nsize > xsize) then
       nwarn=nwarn+1
       write(*,106) me, nsize, xsize
    endif

    !!------------------ alloc #1
    allocate (ua[*])

    nsize = 4
    xsize = xmpf_coarray_allocated_bytes() - xsize
    if (nsize < xsize) then
       nerr=nerr+1
       write(*,105) me, nsize, xsize
    else if (nsize > xsize) then
       nwarn=nwarn+1
       write(*,106) me, nsize, xsize
    endif

    !!------------------ alloc #2
    allocate (va(10)[*])

    nsize = 40
    xsize = xmpf_coarray_allocated_bytes() - xsize
    if (nsize < xsize) then
       nerr=nerr+1
       write(*,105) me, nsize, xsize
    else if (nsize > xsize) then
       nwarn=nwarn+1
       write(*,106) me, nsize, xsize
    endif



    !!------------------ alloc #3
    allocate (xa[*])        !! xmpf_coarray_malloc_generic
                            !!  (desc, xa, 1, sizeof(xa), tag, 0)
    nsize = 56
    xsize = xmpf_coarray_allocated_bytes() - xsize
    if (nsize < xsize) then
       nerr=nerr+1
       write(*,105) me, nsize, xsize
    else if (nsize > xsize) then
       nwarn=nwarn+1
       write(*,106) me, nsize, xsize
    endif

    !!------------------ alloc #4
    allocate (ya(10)[*])    !! xmpf_coarray_malloc_generic
                            !!  (desc, xa, 1, sizeof(ya(lbound(ya,1))), tag, 1, 1, 10)
    nsize = 56
    xsize = xmpf_coarray_allocated_bytes() - xsize
    if (nsize < xsize) then
       nerr=nerr+1
       write(*,105) me, nsize, xsize
    else if (nsize > xsize) then
       nwarn=nwarn+1
       write(*,106) me, nsize, xsize
    endif


    !!------------------ summary
    if (nerr==0 .and. nwarn==0) then 
       print '("[",i0,"] OK")', me
    else
       if (nerr>0) then
          print '("[",i0,"] number of NGs: ",i0)', me, nerr
       endif
       if (nwarn>0) then
          print '("[",i0,"] number of Warnings: ",i0)', me, nwarn
       endif
    end if


105 format("[",i0,"] ERROR: expected ",i0," < allocated ",i0," [bytes]")
106 format("[",i0,"] Warning: expected ",i0," > allocated ",i0," [bytes]")

  end program main
