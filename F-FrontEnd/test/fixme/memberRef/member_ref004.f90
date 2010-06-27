module m
  private
  type s
    integer, dimension(3) :: n
  end type s
  type t
    type(s),pointer :: m
  end type t
  integer, dimension(3) :: a

contains
  subroutine u()
    type(t) :: b
    b%m%n = 1
    a = 2
    a = a + b%m%n + (/1,2,3/)
  end subroutine u
end module m
