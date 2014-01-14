      program main
          integer(kind=4) :: i4
          integer(kind=8) :: i8
          call test(i8)
          call test(0_8)
          call test(i4+1_8)
          call test(i4+0_8)
      contains
          subroutine test(i8)
              integer(kind=8) :: i8
          end subroutine
      end program
