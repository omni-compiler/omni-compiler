program allo3
  include "xmp_lib.h"
  real, allocatable :: a1(:,:)[:],a2(:)[:,:]
  real              :: b1(2,3)[*],b2(8)[4,*]
  real, allocatable :: c1(:,:),c2(:)
  real, allocatable :: b(:)

  write(*,*) "allocated(a1),allocated(c1):",allocated(a1),allocated(c1)
  write(*,*) "allocated(a2),allocated(c2):",allocated(a2),allocated(c2)

  b2(3)[2,1] = 1.234

  allocate (a1(2,3)[*],a2(8)[4,*])
  allocate (c1(2,3),c2(8))

  write(*,*) "allocate (a1(2,3)[*],a2(8)[4,*])"
  write(*,*) "allocate (c1(2,3),c2(8))"
  write(*,*) "allocated(a1),allocated(c1):",allocated(a1),allocated(c1)
  write(*,*) "allocated(a2),allocated(c2):",allocated(a2),allocated(c2)
  write(*,*) "size(a1),size(c1):",size(a1),size(c1)
  write(*,*) "size(a2),size(c2):",size(a2),size(c2)

  a2(3)[2,1] = 1.234
  c2(3) = 1.234

  deallocate (a1,a2)
  deallocate (c1,c2)
  write(*,*) "deallocate (a1,a2)"
  write(*,*) "deallocate (c1,c2)"
  write(*,*) "allocated(a1),allocated(c1):",allocated(a1),allocated(c1)
  write(*,*) "allocated(a2),allocated(c2):",allocated(a2),allocated(c2)

  allocate (a2(3:12)[2,*])
  allocate (c2(3:12))
  write(*,*) "allocate (a2(3:12)[2,*])"
  write(*,*) "allocate (c2(3:12))"
  write(*,*) "allocated(a1),allocated(c1):",allocated(a1),allocated(c1)
  write(*,*) "allocated(a2),allocated(c2):",allocated(a2),allocated(c2)
  write(*,*) "size(a2),size(c2):",size(a2),size(c2)

  if (nerr==0) then 
     print '("[",i0,"] OK")', me
  else
     print '("[",i0,"] number of NGs: ",i0)', me, nerr
  end if


end program allo3

