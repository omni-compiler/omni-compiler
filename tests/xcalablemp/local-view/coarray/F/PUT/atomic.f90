  program main

    integer ii[*], iiok, nn, nnok
    logical la(3,2)[*], laok

    ii=3
    nn=0
    la=.false.
    me=this_image()
    sync all

    nerr=0

    !!------------------- TRY REMOTE-LOCAL
    if (me==2) then
       call atomic_define(ii[1], 333)
    endif
    
    nn = 0
    if (me==1) then
       do while (nn == 0) 
          call atomic_ref(nn, ii)
       enddo
    end if

    !!------------------- TRY LOCAL-REMOTE
    if (me==3) then
       do while (la(2,2))
          call atomic_ref(la(2,2), la(1,2)[2])
       enddo
    endif

    if (me==2) then
       call atomic_define(la(1,2), .true.)
    endif
    
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

    do j=1,2
       do i=1,3
          laok=.false.
          if (i==1 .and. j==2 .and. me==2) laok=.true.
          if (i==2 .and. j==2 .and. me==3) laok=.true.

          if (la(i,j).neqv.laok) then
             nerr=nerr+1
             write(*,101) me, "la", laok, la
          endif
       enddo
    enddo

    !!------------------- SUMMARY OUTPUT
    if (nerr==0) then 
       print '("[",i0,"] OK")', me
    else
       print '("[",i0,"] number of NGs: ",i0)', me, nerr
    end if


100 format("[",i0,"] ",a," should be ",i0," but ",i0,".")
101 format("[",i0,"] ",a," should be ",l1," but ",l1,".")

  end program main

