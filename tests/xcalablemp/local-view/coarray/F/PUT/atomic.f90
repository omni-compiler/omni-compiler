  program main

    integer ii[*], iiok, nn, nnok

    ii=3
    nn=0
    me=this_image()
    sync all

    nerr=0

    !!------------------- TRY REMOTE-LOCAL
    if (me==2) then
       call atomic_define(ii[1], 333)
    endif

    nn = 0
    if (me==1) then
       do while (nn /= 333) 
          call atomic_ref(nn, ii)
       enddo
    end if


    !!------------------- CHECK
    if (me==1) then
       iiok=333
       nnok=333
    else
       iiok=3
       nnok=0
    endif

    if (ii /= iiok) then
       nerr=nerr+1
       write(*,100) me, "ii", iiok, ii
    endif
    if (nn /= nnok) then
       nerr=nerr+1
       write(*,100) me, "nn", nnok, nn
    endif

    !!------------------- SUMMARY OUTPUT
    if (nerr==0) then 
       print '("[",i0,"] OK")', me
    else
       print '("[",i0,"] number of NGs: ",i0)', me, nerr
    end if


100 format("[",i0,"] ",a," should be ",i0," but ",i0,".")
101 format("[",i0,"] ",a," should be ",l2," but ",l2,".")

  end program main

