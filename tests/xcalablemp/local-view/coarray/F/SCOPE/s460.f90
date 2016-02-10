module alloc
include 'xmp_coarray.h'
parameter(lx=100,ly=50)
real*8, allocatable :: a(:,:)[:]
contains
subroutine allocdata
!allocate ( a(lx,ly)[*] )  !! doubtful errCheck() in XMPtransCoarrayRun.run1
!return
end
end module

