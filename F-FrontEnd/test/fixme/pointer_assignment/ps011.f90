      program main
        type t
           integer,pointer,dimension(:,:) :: n
        end type t

        type(t) :: left
        integer,pointer,dimension(:,:) :: right

        left%n => right
      end program main
