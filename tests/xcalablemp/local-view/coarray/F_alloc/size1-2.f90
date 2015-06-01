program main
  include "xmp_coarray.h"
  
!!!!!! not supported !!!!!
!!  interface 
!!     subroutine sub(x,k1,k2)
!!       integer :: x(10)[2,*], k1, k2
!!     end subroutine sub
!!  end interface
  integer :: aaa(20)[*]
  integer :: aab(20)[*]

  call sub(aaa(3),2,1)
end program main

subroutine sub(v2, k1, k2)
  include "xmp_coarray.h"
  integer :: v2(10)[2,*], k1, k2

  integer n(10)
  integer nerr=0

  !! explicit size
  n(1:5) = v2(1:5)[k1,k2]
  if (size(v2(1:5)[k1,k2]).ne.5) nerr=nerr+1

  !! deferred sizes
  n(:) = v2(:)[k1,k2]
  if (size(v2(:)[k1,k2]).ne.10) nerr=nerr+1
  n(1:5) = v2(:5)[k1,k2]
  if (size(v2(:5)[k1,k2]).ne.5) nerr=nerr+1
  n(1:5) = v2(6:)[k1,k2]
  if (size(v2(6:)[k1,k2]).ne.5) nerr=nerr+1
  n(1:4) = v2(::3)[k1,k2]
  if (size(v2(::3)[k1,k2]).ne.4) nerr=nerr+1
  n(1:5) = v2(2::2)[k1,k2]
  if (size(v2(2::2)[k1,k2]).ne.5) nerr=nerr+1
  n(1:4) = v2(:8:2)[k1,k2]
  if (size(v2(:8:2)[k1,k2]).ne.4) nerr=nerr+1

  if (nerr==0) then
     write(*,100) this_image(), "OK"
  else
     write(*,101) this_image(), "NG", nerr
  endif

100 format("[",i0,"] ",a) 
101 format("[",i0,"] ",a," nerr=",i0) 

end subroutine sub

