include 'zmumps_struc.h'
include 'xmp_lib.h'

type ixmp_zmumps_struc
  sequence

  type(xmp_desc) :: idesc
  type(xmp_desc) :: jdesc
  type(xmp_desc) :: adesc
  type(zmumps_struc) :: mumps_par

end type ixmp_zmumps_struc
