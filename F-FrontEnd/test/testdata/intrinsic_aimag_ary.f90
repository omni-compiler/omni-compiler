      program main
          complex a(3), b(3)
          real r
          ! aimag must return real array
          r = dot_product(aimag(a), b)
      end

