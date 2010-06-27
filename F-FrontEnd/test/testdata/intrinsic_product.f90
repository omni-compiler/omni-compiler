      program test_product
        integer,dimension(3,4,5) :: array0
        integer,dimension(5) :: array1

        integer                     i0
        integer,dimension(4,5)   :: i1
        integer,dimension(3,5)   :: i2
        integer,dimension(3,4)   :: i3

        i0 = product(array0)
        i1 = product(array0, 1)
        i2 = product(array0, 2)
        i3 = product(array0, 3)

      end program test_product
