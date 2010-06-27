module m
  private :: f
  interface g
    module procedure f
  end interface
contains
  function f(a)
    integer :: f, a
    f = 1
  end function
  function h()
    integer h
    h = f(1)
  end function
end module m
