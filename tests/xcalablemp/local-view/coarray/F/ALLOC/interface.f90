program main
  interface 
     subroutine sub(x,k1,k2)
       integer :: x(:)[2,*], k1, k2
     end subroutine sub
  end interface
  integer :: a(20)[*]

  call sub(a(1:10))
end program main

subroutine sub(V3, k1, k2)
  include "xmp_coarray.h"
  integer :: V3(:)[2,*], k1, k2

  integer n(10)
  integer nerr=0

  !! explicit size
  n(1:5) = V3(1:5)[k1,k2]
  if (size(V3(1:5)[k1,k2]).ne.5) nerr=nerr+1

  !! deferred sizes
  n(:) = V3(:)[k1,k2]
  if (size(V3(:)[k1,k2]).ne.10) nerr=nerr+1
  n(1:5) = V3(:5)[k1,k2]
  if (size(V3(:5)[k1,k2]).ne.5) nerr=nerr+1
  n(1:5) = V3(6:)[k1,k2]
  if (size(V3(6:)[k1,k2]).ne.5) nerr=nerr+1
  n(1:4) = V3(::3)[k1,k2]
  if (size(V3(::3)[k1,k2]).ne.4) nerr=nerr+1
  n(1:5) = V3(2::2)[k1,k2]
  if (size(V3(2::2)[k1,k2]).ne.5) nerr=nerr+1
  n(1:4) = V3(:8:2)[k1,k2]
  if (size(V3(:8:2)[k1,k2]).ne.4) nerr=nerr+1

end subroutine sub

end
