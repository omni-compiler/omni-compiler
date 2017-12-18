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

        subroutine no_intr_func_1()
          intrinsic :: hoge
          call f(hoge)
        end subroutine no_intr_func_1

        subroutine no_intr_func_2()
          real, intrinsic :: hoge
          call f(hoge)
        end subroutine no_intr_func_2

        subroutine no_intr_func_3()
          intrinsic :: hoge
          real :: hoge
          call f(hoge)
        end subroutine no_intr_func_3

        subroutine no_intr_func_4()
          real :: hoge
          intrinsic :: hoge
          call f(hoge)
        end subroutine no_intr_func_4

        subroutine intr_sub_1()
          intrinsic :: RANDOM_NUMBER
        end subroutine intr_sub_1

        subroutine intr_sub_2()
          intrinsic :: EXIT
        end subroutine intr_sub_2

      end program main
