program main
    integer a
    !$omp parallel default(private)
    a = a + 1
    !$omp end parallel
end program

