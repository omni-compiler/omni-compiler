program main
    integer a
    common /cmn/ a

    !$omp parallel private(a)
    a = 1
    !$omp end parallel
end program
