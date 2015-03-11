program allo1
  include "xmp_lib.h"
  real, allocatable :: a(:)[:,:],a2(:)[:,:]
!!  real, allocatable :: a1(:,:)[*],a2(:)[4,*]   #390
  real, allocatable :: b(:)

  write(*,*) " f:",allocated(a1)," f:",allocated(a2)
  allocate (a1(2,3)[*],a2(8)[4,*])
  write(*,*) " t:",allocated(a1)," t:",allocated(a2)
  deallocate (a1,a2)
  write(*,*) " f:",allocated(a1)," f:",allocated(a2)
  allocate (a2(6)[2,*])
  write(*,*) " f:",allocated(a1)," t:",allocated(a2)

end program allo1

