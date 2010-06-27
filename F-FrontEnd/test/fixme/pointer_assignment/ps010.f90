      program main
        type t
           integer,pointer,dimension(:,:) :: n
        end type t

        type(t) :: left
        integer,target,dimension(4,4) :: right

        right = 4

        left%n => right
      end program main
