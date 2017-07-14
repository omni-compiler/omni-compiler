  program allo_loop
    real, allocatable :: abc(:,:)[:]

    do i=1,10000
       if (mod(i,100)==0) write(*,101) this_image(), i
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
101 format("[",i0,"] executing ",i0,"-th itefation")

    call final_msg(na)

  end program allo_loop


  subroutine final_msg(nerr)
    if (nerr==0) then 
       print '("[",i0,"] OK")', this_image()
    else
       print '("[",i0,"] number of NGs: ",i0)', this_image(), nerr
    end if
    return
  end subroutine final_msg
