      program main
        type structure
           integer :: a
           integer :: b
        end type structure

        type s2
                type(structure) :: c
        end type

        type(structure) s
        type(s2) x

        s%a = 1
        s%b = 2
        x%c%a = 3

        

      end program main
