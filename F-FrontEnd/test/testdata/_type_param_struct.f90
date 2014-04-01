module type_param_struct
  type k
    integer a
    integer b
  end type k
  interface
    pure function foo(x)
      type k
        integer a
        integer b
      end type k
      integer :: foo
      type(k), intent(in) :: x
    end function foo
  end interface
  type(k), parameter :: n = k(1, 4)
  integer(n%a) i
  integer(foo(k(1,4))) j
end module type_param_struct
