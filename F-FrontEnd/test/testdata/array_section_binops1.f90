      program main

      integer :: u(10, 20, 30)
      integer :: v(10, 20, 30, 40)
      integer :: x(10, 20, 30)
      integer :: y(10, 20)
      integer :: z(10)
      integer :: i

      do i = 1, 10
         x(:, :, i) = z(i) * u(:, :, i) * v(:, :, i, i)
      end do

      end program main

      
