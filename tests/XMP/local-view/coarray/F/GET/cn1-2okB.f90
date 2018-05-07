  program char_boundary
!!     include "xmp_coarray.h"
    character(len=8) c5(3)[*]
    character(len=45) c3,val

    me=xmp_node_num()

    !---------------------------- switch on message
!!    call xmpf_coarray_msg(1)

    !---------------------------- initialization
    c3="MerryXmasMerryXmasMerryXmasMerryXmasMerryXmas"
    if (me==1) then
       c5(1)="At last "
       c5(2)="January "
       c5(3)="has gone"
    else if (me==2) then
       c5(1)="Time fli"
       c5(2)="es like "
       c5(3)="an arrow"
    else 
       c5(1)="Oh! my  "
       c5(2)="Budda.  "
       c5(3)=""
    endif
    sync all

    !---------------------------- exec
    if (me==2) then
       c3(10:39)=c5(1)[1]//c5(1)//c5(2)[2]//c5(2)[3]
    end if
    sync all
    
    !---------------------------- check and output
    nerr = 0

    if (me.ne.2) then
       val="MerryXmasMerryXmasMerryXmasMerryXmasMerryXmas"
    else
       val="MerryXmasAt last Time flies like Budda.ryXmas"
    endif

    if (c3.ne.val) then
       nerr=nerr+1
       write(*,101) me
       write(*,*) val
       write(*,102)
       write(*,*) c3
    end if

101 format ("c3 at ",i0," should be: ")
102 format ("but: ")

    if (nerr==0) then 
       print '("[",i0,"] OK")', me
    else
       print '("[",i0,"] number of NGs: ",i0)', me, nerr
    end if

  end program 
