      program main
        real, dimension(5) :: A
        A = (/ -2.0, -1.0, 0.0, 1.0, 2.0 /)
        where (A > 0.0) A = 1.0/A
      end
