  program main

    integer atm[*]
    integer val

    val = 0
    me = this_image()
    sync all

    call atomic_define(atm[mod(me,3)+1], me)
    call wait_sec(1)

    do while (val == 0) 
       call atomic_ref(val, atm)
    end do

    !!--- check
    nerr = 0
    if (val /= modulo(me-2,3)+1) then
       nerr = nerr + 1
       write (*,100) me, "val", modulo(me-2,3)+1, val
    endif

    if (nerr==0) then 
       print '("[",i0,"] OK")', me
    else
       print '("[",i0,"] number of NGs: ",i0)', me, nerr
    end if

100 format("[", i0, "] ", a, " should be ", i0, " but ", i0)

  end program main


  subroutine wait_sec(sec)
    integer sec
    double precision t1, t2
    t1 = xmp_wtime()
    t2 = t1 + sec
    do while (t2 > xmp_wtime())
    enddo
  end subroutine wait_sec

