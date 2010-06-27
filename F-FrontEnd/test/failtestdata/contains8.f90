! same local symbol with no declaration, should refer samve variable.
! compare with testdata/contains4.f90
      program main
        character x
        x = 'A'
        contains
          subroutine func2()
            x = 1
          end subroutine
      end program
