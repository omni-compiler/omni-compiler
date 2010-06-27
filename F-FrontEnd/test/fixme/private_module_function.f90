      module m
        private
        type t
          integer n
        end type t
        interface operator(.in.)
          module procedure g
        !end interface operator(.in.)
        end interface
      contains
        function g(a,b)
          type(t), intent(in) :: a, b
          logical :: g
          g = a%n == b%n
        end function g
      end module m
