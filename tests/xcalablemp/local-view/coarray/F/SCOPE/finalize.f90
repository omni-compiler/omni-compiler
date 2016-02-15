subroutine autosyncall
  include "xmp_coarray.h"
  integer :: mam(19)[2,*]

  mam(9)[2,1] = 1
  if (.true.) goto 10
  return
10 print '("[",i0,"] OK")', this_image()
  stop
  continue
end subroutine autosyncall

program main
  include "xmp_coarray.h"
  call autosyncall
  print '("[",i0,"] NG")', this_image()
end
