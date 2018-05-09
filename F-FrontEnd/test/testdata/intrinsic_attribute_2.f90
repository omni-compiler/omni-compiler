      program main

       contains
        subroutine f(p)
          real :: p
          write (*,*) p(0.1)
        end subroutine f

        subroutine intr_func_1()
          intrinsic :: aint
          call f(aint)
        end subroutine intr_func_1

        subroutine intr_func_2()
          real, intrinsic :: aint
          call f(aint)
        end subroutine intr_func_2

        subroutine intr_func_3()
          intrinsic :: aint
          real :: aint
          call f(aint)
        end subroutine intr_func_3

        subroutine intr_func_4()
          real :: aint
          intrinsic :: aint
          call f(aint)
        end subroutine intr_func_4

      end program main
