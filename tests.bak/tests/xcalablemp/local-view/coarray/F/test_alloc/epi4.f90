program epi4
  include "xmp_coarray.h"
  real, allocatable :: a(:)[:] 

  write(*,*) "1) F ? ", allocated(a)
  call allo
  write(*,*) "2) F ? ", allocated(a)
  call allo
  write(*,*) "3) F ? ", allocated(a)

contains
  subroutine allo
    include "xmp_coarray.h"
    real, allocatable :: a(:)[:]       !! different a
    
    write(*,*) "4) F ? ", allocated(a)
    allocate (a(10)[*])
    write(*,*) "5) T ? ", allocated(a)

  end subroutine allo

end program epi4



