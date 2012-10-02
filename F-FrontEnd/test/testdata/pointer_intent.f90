      subroutine sub(a)
        integer, pointer, intent(out) :: a
      end subroutine sub

      program main
        interface
          subroutine sub(a)
            integer, pointer, intent(out) :: a
          end subroutine sub
        end interface
        integer, pointer :: a
        call sub(a)
      end program main


        
