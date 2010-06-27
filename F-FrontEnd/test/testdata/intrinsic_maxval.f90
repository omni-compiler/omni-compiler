      program test_maxval
        integer,dimension(3,4,5) :: array0
        integer,dimension(5) :: array1

        integer                     i0
        integer,dimension(4,5)   :: i1
        integer,dimension(3,5)   :: i2
        integer,dimension(3,4)   :: i3

        i0 = maxval(array0)
        i1 = maxval(array0, 1)
        i2 = maxval(array0, 2)
        i3 = maxval(array0, 3)

      end program test_maxval
