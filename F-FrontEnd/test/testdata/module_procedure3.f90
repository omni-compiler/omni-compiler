      module mmm 
        interface sub
           module procedure sub1, sub2
        end interface
      contains
        subroutine sub1()
        end subroutine sub1
        subroutine sub2(a)
        end subroutine sub2
      end module mmm
