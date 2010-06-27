      program main
        type t
           integer,pointer :: n
        end type t

        type(t) :: left
        integer,target :: right

        right = 4

        left%n => right
      end program main
