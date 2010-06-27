      program hello_omp
        !$ use omp_lib
        !$ call omp_set_num_thread(4)
        !$OMP parallel
          write(*,*) 'hello'
        !$OMP end parallel  
      end program hello_omp
