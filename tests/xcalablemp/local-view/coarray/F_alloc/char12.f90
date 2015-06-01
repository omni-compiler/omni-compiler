  program allo4
    call sub
  end program allo4

  subroutine sub
    include "xmp_coarray.h"
!!    character(55), allocatable :: s55[:]  !! restriction of OMNI
    character(12), allocatable :: a12(:,:)[:]

    nerr=0

    allocate(a12(2,3)[*])

    if (size(a12,2) /= 3) then
       nerr=nerr+1
       write(*,*) "size(a12,2) must be 3 but", size(a12,2)
    endif

    deallocate (a12)
    if (allocated(a12)) then
       nerr=nerr+1
       write(*,*) "allocated(a12) must be false but", allocated(a12)
    endif

  if (nerr==0) then 
     print '("[",i0,"] OK")', this_image()
  else
     print '("[",i0,"] number of NGs: ",i0)', this_image(), nerr
  end if


  end subroutine sub
