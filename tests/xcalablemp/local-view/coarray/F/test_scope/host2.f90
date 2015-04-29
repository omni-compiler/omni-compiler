    subroutine final_msg(nerr)
      include 'xmp_coarray.h'
      if (nerr==0) then 
         print '("[",i0,"] OK")', this_image()
      else
         print '("[",i0,"] number of NGs: ",i0)', this_image(), nerr
      end if
      return
    end subroutine final_msg

  program host
    include "xmp_coarray.h"
    integer, allocatable :: pap(:,:)[:]
    integer :: mam(19)[*]

    me = this_image()
    mam = me+18

    call party

    contains

      subroutine party
        integer :: kid(3)[2,*]

        kid = (/me,me*10,me*100/)
        
        if (me==2)  mam(3)[1] = kid(2)

        sync all

        nerr = 0
        do i = 1,19
           if (i==3.and.me==1) then
              mmam = 20
           else
              mmam = me+18
           endif
           if (mam(i).ne.mmam) then
              nerr=nerr+1
              write(*,200) me, i, mmam, mam(i)
           endif
        enddo

200     format("[",i0,"] mam(",i0,") should be ",i0," but ",i0,".")

        call final_msg(nerr)

        return
      end subroutine party

    end program host

