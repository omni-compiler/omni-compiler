      program main
        type structure
           integer :: a
           integer :: b
        end type structure

        type s2
                type(structure) :: c
        end type

        type(s2), dimension(10) :: x
        x(5)%c%a = 3
      end program main
