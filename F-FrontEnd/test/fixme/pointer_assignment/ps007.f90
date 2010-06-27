      program main
        type t
           integer, pointer, dimension(:,:) :: n
        end type t
        type s
           integer, pointer,  dimension(:,:) :: n
        end type s

        type(t) :: left
        type(t) :: right

        right%n = 4

        left%n => right%n
      end program main
