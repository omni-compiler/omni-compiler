program allo1
  include "xmp_lib.h"
!!    real, allocatable :: a(:)[:,:]   #390
  real, allocatable :: a1(:,:)[*],a2(:)[4,*]
  real, allocatable :: b(:)

  allocate (a1(2,3)[*],a2(8)[4,*],stat=ierr)
  deallocate (a1,a2,stat=ierr)

end program allo1

