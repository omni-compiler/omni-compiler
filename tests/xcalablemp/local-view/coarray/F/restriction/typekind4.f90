  include "xmp_coarray.h"
  complex(kind=4),allocatable :: a(:)[:]
!!error  complex(kind=16),allocatable :: a(:)[:]
!!abend  double complex,allocatable :: a(:)[:]
!!abend  double precision,allocatable :: a(:)[:]

!!abend
!!  parameter(N=4)
!!  real(kind=N), allocatable :: a(:)[:]

!!error  complex(kind=16) :: b(10)[*]
!!abend  double precision :: b(10)[*]
!!abend  double precision b(10)[*]
!!abend  double complex b(10)[*]
  double complex c(10)               !OK
  double precision d(10)               !OK

  allocate(a(10)[*])
  end
