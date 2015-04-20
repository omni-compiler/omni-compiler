program allo3
  include "xmp_coarray.h"
  real, allocatable :: a1(:,:)[:],a2(:)[:,:]
  real              :: b1(2,3)[*],b2(8)[4,*]
  real, allocatable :: c1(:,:),c2(:)
  real, allocatable :: b(:)

  nerr=0

  if (allocated(a1).neqv.allocated(c1)) nerr=nerr+1
  if (allocated(a2).neqv.allocated(c2)) nerr=nerr+1

  b2(3)[2,1] = 1.234

  allocate (a1(2,3)[*],a2(8)[4,*])
  allocate (c1(2,3),c2(8))

  if (allocated(a1).neqv.allocated(c1)) nerr=nerr+1
  if (allocated(a2).neqv.allocated(c2)) nerr=nerr+1
  if (size(a1).ne.size(c1)) nerr=nerr+1
  if (size(a2).ne.size(c2)) nerr=nerr+1

  a2(3)[2,1] = 1.234
  c2(3) = 1.234

  deallocate (a1,a2)
  deallocate (c1,c2)
  if (allocated(a1).neqv.allocated(c1)) nerr=nerr+1
  if (allocated(a2).neqv.allocated(c2)) nerr=nerr+1

  allocate (a2(3:12)[2,*])
  allocate (c2(3:12))
  if (allocated(a1).neqv.allocated(c1)) nerr=nerr+1
  if (allocated(a2).neqv.allocated(c2)) nerr=nerr+1
  if (size(a2).ne.size(c2)) nerr=nerr+1

  if (nerr==0) then 
     print '("[",i0,"] OK")', this_image()
  else
     print '("[",i0,"] number of NGs: ",i0)', this_image(), nerr
  end if


end program allo3

