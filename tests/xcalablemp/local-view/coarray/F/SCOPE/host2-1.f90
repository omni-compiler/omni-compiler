    subroutine final_msg(nerr)
      include 'xmp_coarray.h'
      if (nerr==0) then 
         print '("[",i0,"] OK")', this_image()
      else
         print '("[",i0,"] number of NGs: ",i0)', this_image(), nerr
      end if
    end subroutine final_msg

  program host
    include "xmp_coarray.h"
    integer, allocatable :: pap(:,:)[:]
    integer :: mam(19)[*]
    integer :: kid(3)[2,*]

    me = this_image()
    mam = me+18
!!    call party
        sync all
        kid = me
        mam(3)[1] = kid(2)

    sync all                     ! unnecessary

    nerr = 0
    do i = 1,19
       if (i==3.and.me==1) then
          mmam = 2
       else
          mmam = me+18
       endif
       if (mam(i).ne.mmam) then
          nerr=nerr+1
          write(*,200) me, i, mmam, mam(i)
       endif
    enddo

200 format("[",i0,"] mam(",i0,") should be ",i0," but ",i0,".")

    call final_msg(nerr)

    contains

      subroutine party
        integer :: kid(3)[2,*]

        sync all
        kid = me
        mam(3)[1] = kid(2)

        sync all

        return
      end subroutine party

    end program host

