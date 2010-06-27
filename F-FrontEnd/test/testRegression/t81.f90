      program main
        implicit none
        external factorial 
        integer res, factorial 
        res = factorial(6)
!       print *, res
      end program
