  module xx
!!     include "xmp_coarray.h"
    integer,save:: aaa[*]
  end module xx

  subroutine zz(c)
    use xx
    integer,save:: bbb[*]
    integer:: c[*]
    aaa[2]=c[1]
    return
  end subroutine zz

  program ma
    use xx
    integer :: ccc[*]
 
    aaa=123
    me=this_image()
    if (me==1) ccc=345
    syncall
    if (me==3) call zz(ccc)
    syncall

    nerr=0
    if (me==2) then
       if (aaa/=345) then
          nerr=1
          write(*,*) "aaa should be 345 but aaa=",aaa
       endif
    else
       if (aaa/=123) then
          nerr=1
          write(*,*) "aaa should be 123 but aaa=",aaa
       endif
    endif

    call final_msg(nerr,me)
  end program ma

  subroutine final_msg(nerr,me)
    if (nerr==0) then 
       print '("[",i0,"] OK")', me
    else
       print '("[",i0,"] number of NGs: ",i0)', me, nerr
    end if
    return
  end subroutine final_msg

    
