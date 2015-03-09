program allo1
  include "xmp_lib.h"
!!    real, allocatable :: a(:)[:]   #390
  real, allocatable :: a(:)[*]
  real, allocatable :: b(:)

  write(*,*) " f:",allocated(a)," f:",allocated(b)
  allocate (a(10)[*])
  write(*,*) " t:",allocated(a)," f:",allocated(b)
  deallocate (a)
  write(*,*) " f:",allocated(a)," f:",allocated(b)

end program allo1

