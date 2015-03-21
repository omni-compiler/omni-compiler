program allo3
  include "xmp_lib.h"
  real, allocatable :: a1(:,:)[:],a2(:)[:,:]
  real              :: b1(2,3)[*],b2(8)[4,*]
  real, allocatable :: c1(:,:),c2(:)
  real, allocatable :: b(:)

  write(*,*) " f:",allocated(a1)," f:",allocated(a2)
  write(*,*) " f:",allocated(c1)," f:",allocated(c2)

  b2(3)[2,1] = 1.234

  allocate (a1(2,3)[*],a2(8)[4,*])
  allocate (c1(2,3),c2(8))
  write(*,*) " t:",allocated(a1)," t:",allocated(a2)
  write(*,*) " t:",allocated(c1)," t:",allocated(c2)

  a2(3)[2,1] = 1.234
  c2(3) = 1.234

  deallocate (a1,a2)
  deallocate (c1,c2)
  write(*,*) " f:",allocated(a1)," f:",allocated(a2)
  write(*,*) " f:",allocated(c1)," f:",allocated(c2)

  allocate (a2(3:12)[2,*])
  allocate (c2(3:12))
  write(*,*) " f:",allocated(a1)," t,10:",allocated(a2),size(a2)
  write(*,*) " f:",allocated(c1)," t,10:",allocated(c2),size(c2)

end program allo3

