      program main
          integer a
          integer omp_get_num_threads
          external omp_get_num_threads
          print *, omp_get_num_threads()
c$omp parallel
c$omp single firstprivate(a)
          print *, "in single"
          a = omp_get_thread_num()
c$omp end single copyprivate(a)
c$omp end parallel
          print *, a
      end program
