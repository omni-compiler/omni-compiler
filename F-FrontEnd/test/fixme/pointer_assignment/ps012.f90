program main
  type t
    integer, pointer, dimension(:,:) :: n
  end type t

  type(t), pointer :: a
  type(t), pointer :: b
  integer, target, dimension(4,4) :: c

  c = 1
  b%n => c
end program main
