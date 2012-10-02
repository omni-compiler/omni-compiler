program pointers
  type t
     integer :: i
  end type t

  type(t), pointer :: u(:) => null()

end program pointers
