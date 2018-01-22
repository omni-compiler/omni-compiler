  program main

    integer atm[*]
    integer val

    val = 0
    me = this_image()
    sync all

    if (me==2) then
       call atomic_define(atm[1], me)
    endif

    call wait_sec(1)

    write(*,100) me
100 format("[", i0, "] succeeded")

  end program main


  subroutine wait_sec(sec)
    integer sec
    double precision t1, t2
    t1 = xmp_wtime()
    t2 = t1 + sec
    do while (t2 > xmp_wtime())
    enddo
  end subroutine wait_sec

