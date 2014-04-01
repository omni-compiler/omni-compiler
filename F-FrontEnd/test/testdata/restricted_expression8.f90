      program foo
          integer m
      contains
          subroutine bar(a,n)
              integer n
              ! host associated object designator is restricted expression
              character(len=n+m) :: a
          end subroutine bar
      end program foo
