module comm2
  implicit none
  integer,parameter :: ndims=3
  integer :: me(3)
  integer :: npe,id
end module comm2

subroutine readparam
!
  use comm2
!
  implicit none
!
  integer, save :: itmp(2,2)[*]
  character(12), save :: size(ndims+1)[*]
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
