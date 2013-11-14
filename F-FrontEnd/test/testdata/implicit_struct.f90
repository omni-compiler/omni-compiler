      module implicit_struct
          type t
          integer i
          end type
      contains
          subroutine test()
              implicit type (t) (x)
          end subroutine
      end module
