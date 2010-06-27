program main
    integer a, b
    common /cmn/ a, b
    !$omp threadprivate(/cmn/)
    a = 0
    b = 1

    !$omp parallel copyin(/cmn/)
    a = a + 1
    b = b + 1
    !$omp end parallel
end program
