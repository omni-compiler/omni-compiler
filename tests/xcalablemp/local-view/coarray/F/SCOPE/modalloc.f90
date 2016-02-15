  module modalloc
    implicit none
    include "xmp_coarray.h"
    real, allocatable, dimension(:,:), codimension[:] :: moda
  end module modalloc

  subroutine suballoc(me)
    implicit none
    use modalloc
    real, allocatable, dimension(:,:), codimension[:] :: suba
    integer i,j,me

    allocate(suba(3,2)[7:*])

    do j=1,2
       do i=1,3
          suba(i,j)=real(i*me)
       enddo
    enddo

    syncall

    do j=1,2
       do i=1,3
          moda(j,i)=suba(i,j)[10-me]
       enddo
    enddo

    return

  end subroutine suballoc

  program main
    implicit none
    use modalloc
    integer me, nerr, i, j
    real val

    allocate(moda(2,3)[10:*])
    me=this_image()

    call suballoc(me)

    syncall

    nerr=0
    do j=1,2
       do i=1,3
          val = real(i*(4-me))
          if (val /= moda(j,i)) then
             nerr=nerr+1
             write(*,200) me, i, j, val, moda(j,i)
!!          else
!!             write(*,200) me, i, j, val, moda(j,i)
          endif
       enddo
    enddo

200 format("[",i0,"] moda(",i0,",",i0,") should be ",f6.2," but ",f6.2,".")

    call final_msg(nerr, me)

  end program main


  subroutine final_msg(nerr,me)
    if (nerr==0) then 
       print '("[",i0,"] OK")', me
    else
       print '("[",i0,"] number of NGs: ",i0)', me, nerr
    end if
    return
  end subroutine final_msg

