program main
  type s
    integer, dimension(3) :: n
  end type s
  type t
    type(s) :: m
  end type t
  integer, dimension(3) :: a

  type(t) :: b
  b%m%n = 1
  a = 2
  a = b%m%n + a
end program main
