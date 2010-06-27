program main
  type t
    integer, dimension(1:3,1:3,1:3) :: n
  end type t
  integer, dimension(1:3,1:3,1:3) :: a

  type(t) :: b
  b%n = 1
  a = 2
  a = b%n + a
end program main
