      module use_in_interface1
      end module use_in_interface1

      module use_in_interface
        interface g
          subroutine sub()
            use use_in_interface1
          end subroutine sub
        end interface
        interface h
          integer function func(i)
            use use_in_interface1
            integer :: i
          end function func
        end interface h
      end module use_in_interface
