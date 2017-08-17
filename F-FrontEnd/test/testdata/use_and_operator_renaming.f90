      module m
        integer :: a
        integer :: c
        integer :: operator
        interface operator(.to.)
          module procedure length
        end interface
        interface operator(.ot.)
          module procedure length
        end interface
       contains
        real function length(a,b)
          real,intent(in):: a,b
          length = b-a
        end function length
      end module m

      subroutine sub1()
        use m, b => a, operator(.toto.) => operator(.to.)
      end subroutine sub1

      subroutine sub2()
        use m, operator(.toto.) => operator(.to.), b => a
      end subroutine sub2

      subroutine sub3()
        use m, b => a, d => c
      end subroutine sub3

      subroutine sub4()
        use m, b => operator, operator => a
      end subroutine sub4

      program main
        use m, operator(.toto.) => operator(.to.) , operator(.otot.) => operator(.ot.)
        real r,s,t
        r=5.0
        s=3.2
        t = r.toto.s
        print *, t
      end program main
