      subroutine sub(m, n)
          real a(m, n)
          complex b(n), c(n)
          a = 2
          b = (1, 1)
          ! retun value of matmul is
          ! complex at this condition
          c = matmul(a, b)
      end

