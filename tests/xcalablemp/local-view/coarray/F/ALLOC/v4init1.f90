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
  if (xmpf_coarray_uses_fjrdma()) then
    nestim = 2*2*4+16
  else
    nestim = 2*2*4+12
  endif
  if (nbytes == nestim) then
     write(*,*) "OK",nbytes,nestim
  else
     write(*,*) "NG",nbytes,nestim
     call exit(1)
  endif
end program main
