      module recursive_struct_type_and_function
        type s
          type(s), pointer :: p
          type(s), pointer :: q
        end type s
      contains
        subroutine subr(arg)
          type(s), pointer :: arg
        end subroutine subr
      end module recursive_struct_type_and_function 
