!! bug #354
!!  include "xmp_coarray.h"
  character(len=2) wbuf1
  character(len=2) wbuf2[*]
  wbuf1=wbuf2[1]
  end
