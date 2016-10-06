      module m1
        private func1
        interface ger
           module procedure func1
        end interface ger
      contains
        integer function func1(arg)
          integer :: arg
          func1 = arg
        end function func1
      end module m1

      module m2
        private func2
        interface ger
           module procedure func2
        end interface ger
      contains
        real function func2(arg)
          real :: arg
          func2 = arg
        end function func2
      end module m2

      module m3
        use m1
        use m2
        integer :: i = 3
      contains
        subroutine sub()
          i = ger(i)
        end subroutine sub
      end module m3

      module m4
        use m1
        interface ger
           module procedure func4
        end interface ger
      contains
        complex function func4(arg)
          complex :: arg
          func4 = arg
        end function func4
        subroutine sub()
          i = ger(i)
        end subroutine sub
      end module m4
