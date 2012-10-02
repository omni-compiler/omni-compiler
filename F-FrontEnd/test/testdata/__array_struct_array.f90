      module array_struct_array
        private 
        type t
          integer :: n
          real*8,dimension(8,8) :: a
        end type t
        type(t),dimension(8),public :: p
      end module array_struct_array
