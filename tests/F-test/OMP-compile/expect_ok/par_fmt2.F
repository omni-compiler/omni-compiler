      program main
      integer i
10    format(I3)
c$omp parallel do
      do i = 1, 3
20    format(I4)
          write(*,10) i
          write(*,20) i
      end do
c$omp end parallel do
      end program

