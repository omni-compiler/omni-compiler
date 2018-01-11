subroutine EX1
!!   include "xmp_coarray.h"
  real :: V1(10,20)[4,*]
  complex(8) :: V2[*], z
  integer n(5)
  integer, allocatable :: V3(:)[:,:]

  V1(1:3,j)[k1,k2] = (/1.0,2.0,3.0/)
  z = V2[1]**2
  n(1:5) = V3(2::2)[k1,k2]
!!  n(1:5) = V3(2:10:2)[k1,k2]

end subroutine EX1

end
