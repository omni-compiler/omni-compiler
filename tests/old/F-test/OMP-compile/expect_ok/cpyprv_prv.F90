program main
    integer a
    integer, external::omp_get_num_threads
    print *, omp_get_num_threads()
    !$omp parallel private(a)
    !$omp single
    print *, "in single"
    a = omp_get_thread_num()
    !$omp end single copyprivate(a)
    print *, a
    !$omp end parallel
end program

