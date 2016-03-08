parameter(lx=10)
include 'xmp_coarray.h'
real, allocatable :: a(:)[:,:]

allocate(a(10)[lx+1,*])

stop
end
