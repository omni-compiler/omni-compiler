module mod1
  CHARACTER(len=5), PARAMETER :: lst(3) = (/ "a", "b", "c" /)
type t
  INTEGER :: len
  LOGICAL :: logis(SIZE(lst))
end type t
end module mod1

program main
  use mod1
  type(t) :: v
  v%len = 1
  v%logis(1) = .TRUE.
end program main
