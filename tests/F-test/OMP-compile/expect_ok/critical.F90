program main
    integer a
    !$omp parallel
    !$omp critical(cc)
    a = 1
    !$omp end critical(cc)
    !$omp end parallel
end program

