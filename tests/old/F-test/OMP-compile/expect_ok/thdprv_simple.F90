program main
    integer,save::a(-2:2)
    integer::i
    integer,external :: omp_get_thread_num
    !$omp threadprivate(a)
    !$omp parallel do private(i)
    do i = -2, 2
        a(i) = i
    end do
    !$omp end parallel do
    !$omp parallel
    print *, omp_get_thread_num(), ":", a
    !$omp end parallel
end program

