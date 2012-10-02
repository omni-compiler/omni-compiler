      module m1
        private func1
        interface operator(.myop.)
           module procedure func1
        end interface
      contains
        integer function func1(a,b)
          integer,intent(in) :: a,b
          func1 = a + b
        end function func1
      end module m1

      module m2
        private func2
        interface operator(.myop.)
           module procedure func2
        end interface
      contains
        real function func2(a,b)
          real,intent(in) :: a,b
          func1 = a + b
        end function func2
      end module m2

      module m3
        use m1
        use m2
        integer :: i = 3
      contains
        subroutine sub()
          i = i .myop. i
        end subroutine sub
      end module m3

      module m4
        use m1
        complex :: i
        interface operator(.myop.)
           module procedure func4
        end interface
      contains
        complex function func4(a,b)
          complex,intent(in) :: a,b
          func1 = a + b
        end function func4
        subroutine sub()
          i = i .myop. i
        end subroutine sub
      end module m4
