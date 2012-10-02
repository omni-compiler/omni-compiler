      module m
        private
        public :: tt
        type t
          integer :: n
        end type t
        type tt
          integer :: p
          type(t) :: q
        end type tt
      contains
        subroutine sub(a)
          type(tt) :: a
          a%q%n = 1
        end subroutine sub
      end module m
