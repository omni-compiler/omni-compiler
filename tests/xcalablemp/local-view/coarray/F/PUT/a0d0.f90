  program test_a2_d0
    real a0[*]
    integer xmp_node_num
    integer nerr

    me = xmp_node_num()   ! == this_image()

    !---------------------------- init
    a0=7.77
    sync all

    !---------------------------- exec
    if (me==1) then
       a0[2]=1.2
    else if (me==2) then
       a0[3]=2.3
    end if

    sync all

    !---------------------------- check and output start
    eps = 0.00001
    nerr = 0
    if (me==2) then
       val=1.2
       if (abs(a0-val)>eps) then
          write(*,100) me,a0,val
          nerr=nerr+1
       end if
    else if (me==3) then
       val=2.3
       if (abs(a0-val)>eps) then
          write(*,100) me,a0,val
          nerr=nerr+1
       end if
    else
       val=7.77
       if (abs(a0-val)>eps) then
          write(*,100) me,a0,val
          nerr=nerr+1
       end if
    end if

    if (nerr==0) then 
       print '("[",i0,"] OK")', me
    else
       print '("[",i0,"] number of NGs: ",i0)', me, nerr
    end if
    !---------------------------- check and output end

100 format ("a0[",i0,"]=",f8.4," should be ",f8.4)

  end program
