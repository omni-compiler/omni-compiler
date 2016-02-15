program epi3
  include "xmp_coarray.h"
  real, allocatable :: a(:)[:] 

  write(*,*) "F ? ", allocated(a)
  call allo(a)
  write(*,*) "T ? ", allocated(a)
  call allo(a)
  write(*,*) "T ? ", allocated(a)

contains
  subroutine allo(a)
    include "xmp_coarray.h"
    real, allocatable :: a(:)[:] 
    real, allocatable :: al(:)[:] 
    
    write(*,*) "F ? ", allocated(al)
    allocate (a(10)[*],al(10)[*])
    write(*,*) "T T ? ", allocated(a), allocated(al)

  end subroutine allo

end program epi3



