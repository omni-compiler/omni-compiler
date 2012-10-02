program pointers
  type t
     integer :: i
  end type t
     type(t), pointer :: v(:)
     type(t), pointer :: w
     v => null()
     w => null()
end program pointers
