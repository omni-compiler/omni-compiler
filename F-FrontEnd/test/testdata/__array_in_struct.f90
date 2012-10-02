      module array_in_struct
        type t
          integer :: n
          real*8,dimension(8) :: array
        end type t
        type(t) :: p
      end module array_in_struct
