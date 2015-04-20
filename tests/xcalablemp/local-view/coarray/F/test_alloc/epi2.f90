program epi2
  include "xmp_coarray.h"
  real, allocatable :: a(:)[:] 

  write(*,*) "1) F ? ", allocated(a)
  call allo
  write(*,*) "2) T ? ", allocated(a)
  call allo
  write(*,*) "3) T ? ", allocated(a)

  !! a should be dealloc here.
contains
  subroutine allo
    include "xmp_coarray.h"
    
    write(*,*) "4) F-T ? ", allocated(a)
    allocate (a(10)[*])
    write(*,*) "5) T ? ", allocated(a)

    !! a should not be dealloc here.
  end subroutine allo

end program epi2



