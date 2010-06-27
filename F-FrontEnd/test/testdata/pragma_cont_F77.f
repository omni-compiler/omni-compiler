      integer a
c$omp parallel
c$omp+private(a)
      print *,a
c$omp end parallel
      end

