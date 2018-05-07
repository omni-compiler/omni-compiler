!!   include "xmp_coarray.h"
  integer ierr
  character(100) msg
  integer dmy[2,*]
  double precision dmy2(2,1)[*]
  syncall
!  sync all(stat=ierr)
!  syncimages(errmsg=msg,stat=ierr)
!  syncimages(stat=ierr)
!  sync images
  if (this_image()==1) then
     sync images(2)
     sync images((/2,3/))
  else if (this_image()==2) then
     sync images(1)
     sync images((/1,3/))
  else if (this_image()==3) then
     sync images((/2,1/))
  endif
!  sync images(*)             ! Not implement xmp_sync_images_all()
!  sync images(1,stat=ierr)
!  sync images((/2,3/),stat=m)
!  lock
!  unlock
!  syncmemory(errmsg=msg)
  sync memory
!  sync memory(stat=ierr,errmsg=msg)
!  critical
!  end critical
!  errorstop
!  error stop
  end
