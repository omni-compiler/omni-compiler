      module m
        use extfile_generic_procedure1
        use extfile_generic_procedure2
        integer :: i
      contains
        subroutine sub()
          i = g(i)
        end subroutine
      end module m
