module elemental_func
  implicit none
contains
  elemental real function square(x)
    real, intent(in) :: x
    square = x*x
  end function
end module
