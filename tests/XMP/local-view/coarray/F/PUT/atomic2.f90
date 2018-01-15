  program main

    integer ii[*]
    integer ia(3,2)[*]

    ii=3
    ia=4
    nerr=0
    sync all

    me = this_image()

    if (me==3) then
       nval=4
       do while (nval<100)
          call atomic_ref(nval,ia(2,1))
       enddo
       if (nval.ne.444) then
          nerr=nerr+1
          write(*,100) me,"nval",444,nval
       endif
    endif

    if (me==1) then
       nval = 3
       do while (nval<100)
          call atomic_ref(nval, ii[2])
       enddo
       if (nval.ne.333) then
          nerr=nerr+1
          write(*,100) me,"nval",333,nval
       endif
    endif

    if (me==2) then
       call atomic_define(ii, 333)
       call atomic_define(ia(2,1)[3], 444)
    endif
    sync all
    
    write(*,101) me, "OK completed."

100 format("[",i0,"] ",a," should be ",i0," but ",i0)
101 format("[",i0,"] ",a)

  end program main

