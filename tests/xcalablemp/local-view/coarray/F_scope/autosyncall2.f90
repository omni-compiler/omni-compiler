subroutine autosyncall1
  include "xmp_coarray.h"
  integer :: mam(19)[2,*]

  mam(9)[2,1] = 1
  if (.true.) return
  goto 10
  return
10 stop
  continue
end subroutine autosyncall1

program main
end
