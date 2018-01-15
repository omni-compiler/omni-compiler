module comm3
  implicit none
  integer,parameter :: ndims=3
  integer,parameter :: len=4
  integer :: npe,id
end module comm3

subroutine readparam
!
  use comm3
!
  implicit none
!
  integer, save :: itmp(2,2)[*]
  character(len=len*3), save :: size(ndims+1)[*]
!
end subroutine readparam

program main
  call readparam
  nbytes = xmpf_coarray_allocated_bytes()
  nestim = 2*2*4 + (3+1)*12
  if (nbytes == nestim) then
     write(*,*) "OK",nbytes,nestim
  else
     write(*,*) "NG",nbytes,nestim
  endif
end program main
