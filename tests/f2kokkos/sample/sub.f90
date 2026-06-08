subroutine sub(a, x, y)

  use iso_fortran_env, only : int64
  
  integer, parameter :: N1 = 4096
  integer, parameter :: N2 = 4096

  real :: a
  real :: x(N1,N2), y(N1,N2)

  integer(int64) :: t0, t1, rate

  call system_clock(t0, rate)

  do k=1, 10000
  
!$xmp parallel_for (x,y,a) on (i,j)
  do j=1, N2
     do i=1, N1
        y(i,j) = y(i,j) + a * x(i,j)
     end do
  end do

end do
  
  call system_clock(t1)

  write(*,*) "elapsed time =", real(t1 - t0, 8) / real(rate, 8), t1-t0, rate
  
end subroutine sub
