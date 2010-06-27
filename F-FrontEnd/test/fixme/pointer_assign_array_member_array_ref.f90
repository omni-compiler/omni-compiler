program main
  type t
    integer,dimension(8) :: n
  end type
  type(t), pointer, dimension(:,:,:) :: b
  integer, pointer, dimension(:) :: a
  a => b(1:8,1,1)%n(1)
end program main
