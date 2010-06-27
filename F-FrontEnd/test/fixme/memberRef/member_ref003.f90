program main
  type s
    integer, dimension(3) :: n
  end type s
  type t
    type(s),pointer :: m
  end type t
  integer, dimension(3) :: a

  type(t) :: b
  b%m%n = 1
  a = 2
  a = a + b%m%n + (/1,2,3/)  ! NG
!  a = b%m%n + (/1,2,3/) + a ! OK
!  a = a + b%m%n             ! OK
!  a = b%m%n + (/1,2,3/)     ! OK
!  a = a + (/1,2,3/)     ! OK
end program main
