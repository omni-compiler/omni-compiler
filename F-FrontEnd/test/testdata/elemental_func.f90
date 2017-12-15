module elemental_func
  implicit none
contains
  elemental real function square(x)
    real, intent(in) :: x
    square = x*x
  end function
end module

program main
  use elemental_func
  print *, square((/1.0, 2.0, 3.0/))
end program main
