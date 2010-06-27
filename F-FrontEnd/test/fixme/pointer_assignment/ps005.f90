      program main
        type t
           integer, pointer :: n
        end type t

        type(t) :: left
        type(t) :: right

        right%n = 4

        left%n => right%n
      end program main
