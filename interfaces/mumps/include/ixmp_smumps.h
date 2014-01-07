include 'smumps_struc.h'
include 'xmp_lib.h'

type ixmp_smumps_struc
  sequence

  type(xmp_desc) :: idesc
  type(xmp_desc) :: jdesc
  type(xmp_desc) :: adesc
  type(smumps_struc) :: mumps_par

end type ixmp_smumps_struc
