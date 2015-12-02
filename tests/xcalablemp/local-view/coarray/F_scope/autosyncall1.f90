subroutine autosyncall(n1,n2)
  include "xmp_coarray.h"
  integer, allocatable :: pap(:,:)[:]
  integer :: mam(19)[2,*]

  mam(9)[2,1] = 1
  allocate(pap(n1,n2)[*])
  pap(n1/2+1,n1) = mam(9)[2,1]
  if (.true.) return

end subroutine autosyncall

program main
  include "xmp_coarray.h"
  call autosyncall(1,1)
  call autosyncall(13,15)
  call autosyncall(5,2)
  call autosyncall(2,2)
  print '("[",i0,"] OK")', this_image()
end

