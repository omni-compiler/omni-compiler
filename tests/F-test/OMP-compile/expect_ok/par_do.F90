program main
    integer::i, n
    integer, external::omp_get_thread_num

    !$omp parallel do schedule(static)
    do i = 1, 9, 1
        n = omp_get_thread_num()
        print *, n, ":", i
    end do
    !$omp end parallel do
end program
