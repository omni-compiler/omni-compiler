      program test_abs
        integer :: i = -1
        real :: x = -1.e0
        double precision :: y = -2.e0
        complex :: z = (-1.e0,0.e0)
        i = abs(i)
        x = abs(x)
        x = abs(z)
        x = cabs(z)
        y = dabs(y)
        i = iabs(i)
      end program test_abs
