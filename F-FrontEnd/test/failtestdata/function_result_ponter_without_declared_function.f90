      module mod
        real,target :: tmp
       contains
        subroutine sub()
          real a
          a = 3.0
          print *, f(a)
          f(a) = 6.0
          print *, f(a)
        end subroutine sub
      end module mod
