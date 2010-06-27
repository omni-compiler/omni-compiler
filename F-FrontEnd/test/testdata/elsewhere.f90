      program main
        real, dimension(5) :: A

        where (A > 0.0)
           A = 1.0 / A
        elsewhere
           A = 2 * A
        end where

      end program main
