      subroutine s
100   format("test")
c$omp parallel
      write(*, 100)
c$omp end parallel
      end subroutine
