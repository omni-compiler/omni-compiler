  subroutine sub1
    continue
  end subroutine

  subroutine sub2
    real, allocatable:: a(:,:), b(:,:)
    allocate(b(10,20))
    allocate(a(3,3))
    deallocate(b)
  end subroutine

  program main
    integer size1, size2, size3,me
    me = this_image()
    size1 = xmpf_coarray_malloc_bytes()
    call sub1
    size2 = xmpf_coarray_malloc_bytes()
    call sub2
    size3 = xmpf_coarray_malloc_bytes()

!!--- check
    nerr=0
    if (size1.ne.size2) then
       nerr=nerr+1
       write(*,100) me, "sub1", size1, size2
    endif
    if (size2.ne.size3) then
       nerr=nerr+1
       write(*,100) me, "sub2", size2, size3
    endif

  if (nerr==0) then 
     print '("[",i0,"] OK")', this_image()
  else
     print '("[",i0,"] number of NGs: ",i0)',  this_image(), nerr
     stop 1
  end if

100 format("[",i0,"] NG: guess memory leak in ", a, ". ",i0," bytes and ",i0," bytes.")

  end program main

