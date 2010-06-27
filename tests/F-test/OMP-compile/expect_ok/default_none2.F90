      subroutine s(n, a)
          integer n
          integer a(n)
!$omp parallel default(none) private(a)
          a = 1
!$omp end parallel
      end

