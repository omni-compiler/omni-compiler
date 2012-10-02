      module struct_typed
        type base_class
          integer :: i
        end type base_class
      contains
        type(base_class) function f()
          f%i = 1
        end function f
      end module struct_typed
