      module m1
      contains
        subroutine sub()
        end subroutine sub
      end module m1

      module m2
      contains
        subroutine sub()
        end subroutine sub
      end module m2

      program main
        use m1
        use m2
        call sub()
      end program main

