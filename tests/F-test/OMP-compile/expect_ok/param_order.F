      subroutine sub
          implicit none
          integer a, b, c
          parameter(c = 1)
          parameter(b = c)
          parameter(a = b)
          integer d
          integer e(a + b + c)
          common /cmn/ e
c$omp parallel
          d = a + b + c
c$omp end parallel
      end subroutine
      
