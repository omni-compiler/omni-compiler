program main
    integer a
    a = 1
    !$omp parallel firstprivate(a)
        print *, a
    !$omp end parallel
end program
