program main
    integer a
    !$omp parallel
    !$omp critical(cc)
    !$omp critical(dd)
    a = 1
    !$omp end critical(dd)
    !$omp end critical(cc)
    !$omp end parallel
end program

