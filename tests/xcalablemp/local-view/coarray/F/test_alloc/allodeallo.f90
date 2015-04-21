program epi2
  include "xmp_coarray.h"
  real, allocatable :: a(:)[:] 

  nerr=0
  if (allocated(a)) then
     nerr=nerr+1
     write(*,*) "1) F ? ", allocated(a)
  endif

  call allo

  if (.not.allocated(a)) then
     nerr=nerr+1
     write(*,*) "1) F ? ", allocated(a)
  endif

  call deallo

  if (allocated(a)) then
     nerr=nerr+1
     write(*,*) "1) F ? ", allocated(a)
  endif

  if (nerr==0) then 
     print '("[",i0,"] OK")', this_image()
  else
     print '("[",i0,"] number of NGs: ",i0)', this_image(), nerr
  end if

contains
  subroutine allo
    include "xmp_coarray.h"
    allocate (a(10)[*])
  end subroutine allo

  subroutine deallo
    include "xmp_coarray.h"
    deallocate (a)
  end subroutine deallo

end program epi2



