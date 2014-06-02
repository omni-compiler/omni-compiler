program main
    integer a
    integer, external::omp_get_thread_num

    !$omp parallel default(private)
    !$omp single
    a = omp_get_thread_num()
    !$omp end single copyprivate(a)
    print *, a
    !$omp end parallel

end program

