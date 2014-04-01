      module foo
          integer m
      end module foo
      subroutine bar(a,n)
          use foo
          integer n
          ! use associated object designator is restricted expression
          character(len=n+m) :: a
      end subroutine bar
