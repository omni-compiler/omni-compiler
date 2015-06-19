  module mmm
    include "xmp_coarray.h"
    integer :: pot(0:99)[2,*]
    integer me
  end module mmm

  program mod1
    use mmm
    integer :: pom(0:99)[*]

!-------------------------------- init
    me = this_image()
    nerr = 0

    do j=0,99
       pom(j) = me*j
       pot(j) = me+j
    enddo

    syncall

!-------------------------------- exec
    if (me==2) pot[1,2] = pom

    syncall

!-------------------------------- check
    do j=0,99
       if (me==3) then
          nok = 2*j
       else
          nok = me+j
       endif
       if (pot(j).ne.nok) then
          nerr=nerr+1
          write(*,100) me, j, nok, pot(j)
       endif
    enddo

100 format("[",i0,"] pot(",i0,") should be ",i0," but ",i0)

    call final_msg(nerr,me)

  end program mod1


  subroutine final_msg(nerr, me)
    if (nerr==0) then 
       print '("[",i0,"] OK")', me
    else
       print '("[",i0,"] number of NGs: ",i0)', me, nerr
    end if
  end subroutine final_msg
