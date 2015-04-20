program allo_stat
  include "xmp_coarray.h"
  real, allocatable :: a1(:,:)[:],a2(:)[:,:]
  real, allocatable :: b(:)

  allocate (a1(2,3)[*],a2(8)[4,*],stat=ierr)
  deallocate (a1,a2,stat=ierr)

end program

