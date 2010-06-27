      subroutine s()
      end subroutine s

      program main
        external s
      contains
        subroutine t()
          call s()
        end subroutine t
      end program main
