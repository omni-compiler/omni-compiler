  program host
    include "xmp_coarray.h"
    integer, allocatable :: pap(:,:)[:]
    integer :: mam(19)[2,*]

    me = this_image()
    mam = me+18

    if (me==1)  call party
    sync all

    nerr = 0
    do i = 1,19
       if (i==3.and.me==1) then
          mmam = 2*100+1
       else
          mmam = me+18
       endif
       if (mam(i).ne.mmam) then
          nerr=nerr+1
          write(*,100) me, i, mmam, mam(i)
       endif
    enddo

100 format("[",i0,"] mam(",i0,") should be ",i0," but ",i0,".")

    call final_msg(nerr)

    contains

      subroutine party
        integer :: kid(3)[2,*]

        do i=1,3
           kid(i) = i*100+me
        enddo
        mam(3)[1,1] = kid(2)

        return
      end subroutine party

    end program host

    subroutine final_msg(nerr)
      include 'xmp_coarray.h'
      if (nerr==0) then 
         print '("[",i0,"] OK")', this_image()
      else
         print '("[",i0,"] number of NGs: ",i0)', this_image(), nerr
      end if
      return
    end subroutine final_msg
