      program hello_omp
c$OMP parallel
          write(*,*) 'hello'
c$OMP end parallel  
      end
