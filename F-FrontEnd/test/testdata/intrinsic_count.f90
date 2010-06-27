      program test_count
        logical,dimension(3,4,5) :: array0

        integer                     i0
        integer,dimension(4,5)   :: i1
        integer,dimension(3,5)   :: i2
        integer,dimension(3,4)   :: i3

        i0 = count(array0)
        i1 = count(array0, 1)
        i2 = count(array0, 2)
        i3 = count(array0, 3)
      end program test_count
