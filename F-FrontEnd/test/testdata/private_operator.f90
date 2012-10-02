      module m
        private :: operator(+)
        public :: operator(-)
        type t
          integer n
        end type t
        interface operator(+)
          module procedure f
        end interface operator(+)
        interface operator(-)
          module procedure f
        end interface operator(-)
      contains
        function f(a,b)
          type(t) :: f
          type(t),intent(in) :: a, b
          f%n = a%n + b%n
        end function f
      end module m
