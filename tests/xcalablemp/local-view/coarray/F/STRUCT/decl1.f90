  program get1
    integer,parameter:: z=2

    type g1
       integer*8 a(10)
       real*8    r(10)
    end type g1

    real, save :: nr1(z+1:z+3,-5:19), cr1(z+1:z+3,-5:19)[*]
!!    type(g1), save :: ns1, cs1[*]
    type(g1), save :: na1(z+1:z+3,-5:19), ns1(z+1:z+3,-5:19)[*]

    if (nerr==0) then 
       print '("result[",i0,"] OK")', me
    else
       print '("result[",i0,"] number of NGs: ",i0)', me, nerr
    end if
    !---------------------------- check and output end

  end program
