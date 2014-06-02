program main
    integer a
    !$omp parallel default(shared)
    a = a + 1
    !$omp end parallel
end program

