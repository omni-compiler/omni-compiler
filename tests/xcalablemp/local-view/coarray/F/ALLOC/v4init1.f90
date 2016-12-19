subroutine readparam
!
  implicit none
!
  integer, save :: itmp(2,2)[*]
  character(12), save :: size(1)[*]
!
end subroutine readparam

program main
  call readparam
  nbytes = xmpf_coarray_allocated_bytes()
  nestim = 2*2*4+12
  if (nbytes == nestim) then
     write(*,*) "OK",nbytes,nestim
  else
     write(*,*) "NG",nbytes,nestim
  endif
end program main
