! same local symbol is treat as same if not defined in child
      program main
        real x
        x = 1.
        contains
          subroutine func2()
            x = 2.
          end subroutine
      end program
