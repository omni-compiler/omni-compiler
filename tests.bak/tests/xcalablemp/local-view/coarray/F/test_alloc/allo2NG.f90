program allo2
  include "xmp_coarray.h"
  real, allocatable :: a(:)[:] 
!!  real, allocatable :: a(:)[*]  #390

  write(*,*) "f", allocated(a)
  call allo
  write(*,*) "t", allocated(a)
  write(*,*) "size(a)=10", size(a)
  call reallo
  write(*,*) "t", allocated(a)
  write(*,*) "size(a)=100", size(a)
  stop

contains
  subroutine allo
    write(*,*) "f", allocated(a)
    allocate (a(10))
    write(*,*) "t", allocated(a)
  end subroutine allo
      
  subroutine reallo
    deallocate (a)
    write(*,*) "f", allocated(a)
    allocate (a(100))
  end subroutine reallo
      
end program allo2
