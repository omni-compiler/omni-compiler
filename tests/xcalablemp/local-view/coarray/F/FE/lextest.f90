  include "xmp_coarray.h"
  integer ierr
  character(100) msg
  syncall
  sync all(stat=ierr)
!  syncimages(errmsg=msg,stat=ierr)
!  syncimages(stat=ierr)
!  sync images
  sync images(1)
  sync images(1,stat=ierr)
  sync images((/2,3/),stat=m)
  lock
  unlock
!  syncmemory(errmsg=msg)
  sync memory
  sync memory(stat=ierr,errmsg=msg)
  critical
  end critical
  critical
  endcritical
  errorstop
  error stop
  end
