subroutine sub()
  complex :: kk
  intrinsic :: imag
  intrinsic :: exit

  write(*,*) imag(kk)
  call exit()
end subroutine sub

program test
  implicit none
  complex :: kk
  intrinsic :: imag
  intrinsic :: exit

  write(*,*) imag(kk)
  call exit()
end program test
