      program main
        character (len=4) :: a, b
        character (len=3) :: c (2)
        equivalence (a, c(1)), (b, c(2))
      end
