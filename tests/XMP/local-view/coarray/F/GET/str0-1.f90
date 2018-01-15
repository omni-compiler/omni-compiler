  program gettest_strct_d0
!!     include "xmp_coarray.h"
    type str
       integer(4) nn
       real(4) rr
    end type str

    type(str) :: sg[*]
!!    type(str),save :: sg[*]
    type(str) :: sl

    me=xmp_node_num()

    !---------------------------- initialization
    sl%nn=me*10
    sl%rr=me*10.0
    sync all

    !---------------------------- exec
    if (me==2) then
       sl%nn=sg[1]%nn
       sl%rr=sg[3]%rr
    endif
    sync all
    
    !---------------------------- check and output
    nerr = 0
    eps=0.00000001
    if (me==2) then
       nn=1*10
       rr=3*10
    else
       nn=me*10
       rr=me*10.0
    endif

    if (sl%nn /= nn) then
       print '("sl[",i0,"]%nn=",i0," should be ",i0)', me,sl%nn,nn
       nerr=nerr+1
    end if
    if (sl%rr - rr > eps) then
       print '("sl[",i0,"]%rr=",f0," should be ",f0)', me,sl%rr,rr
       nerr=nerr+1
    end if

    if (nerr==0) then 
       print '("[",i0,"] OK")', me
    else
       print '("[",i0,"] number of NGs: ",i0)', me, nerr
    end if

  end program gettest_strct_d0
