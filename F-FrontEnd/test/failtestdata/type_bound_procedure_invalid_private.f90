      module m
        type t
         integer :: i
        contains
         procedure :: method => f
         PRIVATE
         procedure :: method2 => f
        end type t
      contains
        function f(this)
          class(t) :: this
          this%i = 1
        end function f
      end module m

