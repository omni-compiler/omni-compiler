      recursive function factorial(n) result(res)
        implicit none
        integer n, res
        if (n < 1) then
          res = 1
        else
          res = n * factorial(n - 1)
        end if
      end function

      program main
        implicit none
        external factorial 
        integer res, factorial 
        res = factorial(6)
!       print *, res
      end program
