include 'cmumps_struc.h'
include 'xmp_lib.h'

type ixmp_cmumps_struc
  sequence

  type(xmp_desc) :: idesc
  type(xmp_desc) :: jdesc
  type(xmp_desc) :: adesc
  type(cmumps_struc) :: mumps_par

end type ixmp_cmumps_struc
