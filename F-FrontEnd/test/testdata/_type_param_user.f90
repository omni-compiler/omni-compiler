module type_param_user
  interface operator(.hoge.)
    module procedure bin_ope
  end interface operator(.hoge.)
  integer(1 .hoge. 3) :: i
  contains
    pure function bin_ope(a, b)
      integer(kind=4), intent(in) :: a, b
      integer(kind=4) :: bin_ope
      bin_ope = a + b
    end function
end module type_param_user
