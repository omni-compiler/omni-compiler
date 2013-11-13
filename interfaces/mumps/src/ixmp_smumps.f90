subroutine ixmp_smumps(id, dirn, djcn, da)

  include "smumps_struc.h"
  type(smumps_struc) id
  integer*8 dirn,djcn,da
  integer irn_size,jcn_size,a_size

  ierr = xmp_array_lsize(dirn,1,irn_size)
  ierr = xmp_array_lsize(djcn,1,jcn_size)
  ierr = xmp_array_lsize(da,1,a_size)
  id%nz_loc=irn_size

  if (id%job == 1 .or. id%job==2  .or. id%job==4 .or. id%job==5 .or. id%job==6) then
    if (associated(id%irn_loc)) then
    else
      allocate(id%irn_loc (irn_size))
      ierr=ixmp_array_icopy(dirn, id%irn_loc)
    endif
    if (associated(id%jcn_loc)) then
    else
      allocate(id%jcn_loc (jcn_size))
      ierr=ixmp_array_icopy(djcn, id%jcn_loc)
    endif
    if (associated(id%a_loc)) then
    else
      allocate(id%a_loc (a_size))
      ierr=ixmp_array_scopy(da, id%a_loc)
    endif

  endif

  if (id%job == -2) then
    if (associated(id%irn_loc)) then
      deallocate(id%irn_loc)
    endif
    if (associated(id%jcn_loc)) then
      deallocate(id%jcn_loc)
    endif
    if (associated(id%a_loc)) then
      deallocate(id%a_loc)
    endif
  end if

  call smumps(id)

  return
  end
