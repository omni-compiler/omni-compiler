module m
  type t
    integer t
  end type
end module m

module n
  use m
  type(t) :: a
contains
  subroutine s()
    type(t) :: b
  end subroutine s
end module n
