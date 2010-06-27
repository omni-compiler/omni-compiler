program main
    integer, save::a(2)
    integer, external::omp_get_thread_num
    !$omp parallel
    a(omp_get_thread_num() + 1) = 1
    !$omp end parallel
    print *,a
end program 

