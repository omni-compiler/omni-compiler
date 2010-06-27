! same local symbol is treat as different in parent and child
      program main
        real x
        x = 1.
        contains
          subroutine func2()
            character x
            x = 'A'
          end subroutine
      end program
