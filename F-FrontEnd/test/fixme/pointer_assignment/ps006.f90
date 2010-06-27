      program main
        type t
           integer, pointer :: n
        end type t
        type s
           integer, pointer :: n
        end type s

        type(t) :: left
        type(t) :: right

        left%n => right%n
      end program main
