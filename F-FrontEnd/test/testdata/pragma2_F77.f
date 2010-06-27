      A=1;B=2
C$    A=100;B=200
C$OMP parallel private(A)
c$OMP parallel private(B)
      print *,A
!$OMP end parallel
!$omp end parallel
      end

