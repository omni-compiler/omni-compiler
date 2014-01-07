subroutine ixmp_zmumps(id)

  include "ixmp_zmumps.h"
  type(ixmp_zmumps_struc) id
  integer irn_size,jcn_size,a_size

  if ((id%mumps_par%icntl(18)==2 .or. id%mumps_par%icntl(18)==3) .and. &
      (id%mumps_par%job == 1 .or. id%mumps_par%job==2  .or. &
       id%mumps_par%job==4 .or. id%mumps_par%job==5 .or. &
       id%mumps_par%job==6)) then

    ierr = xmp_array_lsize(id%idesc,1,irn_size)
    ierr = xmp_array_lsize(id%jdesc,1,jcn_size)
    ierr = xmp_array_lsize(id%adesc,1,a_size)
    id%mumps_par%nz_loc=irn_size

    if (associated(id%mumps_par%irn_loc)) then
    else
      allocate(id%mumps_par%irn_loc (irn_size))
    endif
    ierr=ixmp_array_icopy(id%idesc, id%mumps_par%irn_loc)
    if (associated(id%mumps_par%jcn_loc)) then
    else
      allocate(id%mumps_par%jcn_loc (jcn_size))
    endif
    ierr=ixmp_array_icopy(id%jdesc, id%mumps_par%jcn_loc)
    if (associated(id%mumps_par%a_loc)) then
    else
      allocate(id%mumps_par%a_loc (a_size))
    endif
    ierr=ixmp_array_zcopy(id%adesc, id%mumps_par%a_loc)

  endif

  if (id%mumps_par%job == -2) then
    if (associated(id%mumps_par%irn_loc)) then
      deallocate(id%mumps_par%irn_loc)
    endif
    if (associated(id%mumps_par%jcn_loc)) then
      deallocate(id%mumps_par%jcn_loc)
    endif
    if (associated(id%mumps_par%a_loc)) then
      deallocate(id%mumps_par%a_loc)
    endif
  end if

  call zmumps(id%mumps_par)

  return
  end
