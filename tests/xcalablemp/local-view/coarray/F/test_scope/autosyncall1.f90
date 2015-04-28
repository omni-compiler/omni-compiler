subroutine autosyncall1
  include "xmp_coarray.h"
  integer, allocatable :: pap(:,:)[:]
  integer :: mam(19)[2,*]

  mam(9)[2,1] = 1
  if (.true.) return

end subroutine autosyncall1

program main
end

