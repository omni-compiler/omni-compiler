      module recursive_struct_type
        type s
          type(s), pointer :: next1
          type(s), pointer :: next2
        end type
        type(s), pointer :: q
      end module recursive_struct_type
