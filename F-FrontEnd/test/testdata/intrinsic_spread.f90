      program test_count
        integer,dimension(7,7,7) :: array0

        integer,dimension(1,7,7,7) :: i1
        integer,dimension(7,2,7,7) :: i2
        integer,dimension(7,7,3,7) :: i3
        integer,dimension(7,7,7,4) :: i4

        i1 = spread(array0, 1, 1)
        i2 = spread(array0, 2, 2)
        i3 = spread(array0, 3, 3)
        i4 = spread(array0, 4, 4)

      end program test_count
