      program test_int
        integer :: i = 42
        real*4  :: r4 = 3.9
        real*8  :: r8 = 3.9
        complex :: z = (-3.7, 1.0)
        print *, int(i)
        print *, int(z), int(z,8)
        print *, ifix(r4)
        print *, idint(r8)
      end program
