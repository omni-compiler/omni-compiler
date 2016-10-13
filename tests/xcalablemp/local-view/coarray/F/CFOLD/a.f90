module param
integer, parameter :: lx=100
end module

!! include 'xmp_coarray.h'
use param
real*8 :: a(lx)[*]

end
