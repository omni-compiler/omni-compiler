program epi1
  include "xmp_coarray.h"
  real, allocatable :: a(:)[:] 

  interface
     subroutine allo(a)
       include "xmp_coarray.h"
       real, allocatable :: a(:)[:] 
       real, allocatable :: al(:)[:] 
     end subroutine allo
  end interface

  write(*,*) "F ? ", allocated(a)
  call allo
  write(*,*) "T ? ", allocated(a)
  call allo
  write(*,*) "T ? ", allocated(a)

end program epi1


subroutine allo(a)
  include "xmp_coarray.h"
  real, allocatable :: a(:)[:] 
  real, allocatable :: al(:)[:] 
    
  write(*,*) "F ? ", allocated(al)
  allocate (a(10)[*],al(10)[*])
  write(*,*) "T T ? ", allocated(a), allocated(al)

end subroutine allo


