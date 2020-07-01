  integer function pepe()
    pepe=3
    return 
  end function pepe

!!   include "xmp_coarray.h"
  integer aa(10)[*]
  integer pepe

  me = xmp_node_num()

!--------------------- init
  do i=1,10
     if (me==3) then
        aa(i)=10-i
     else
        aa(i)=-i*100
     end if
  enddo

  sync all

!--------------------- exec

  aa(pepe():aa(6)[3])[1] = (/555,666/)

  sync all

!--------------------- check
  nerr=0

  do i=1,10
     if (me==1.and.i==3) then
        ival = 555
     else if (me==1.and.i==4) then
        ival = 666
     else if (me==3) then
        ival=10-i
     else
        ival=-i*100
     end if

     if (aa(i).ne.ival) then 
        write(*,101) i,me,aa(i),ival
        nerr=nerr+1
     endif
  enddo

    if (nerr==0) then 
       print '("[",i0,"] OK")', me
    else
       print '("[",i0,"] number of NGs: ",i0)', me, nerr
       call exit(1)
    end if

101 format ("aa(",i0,")[",i0,"]=",i0," should be ",i0)


  end
