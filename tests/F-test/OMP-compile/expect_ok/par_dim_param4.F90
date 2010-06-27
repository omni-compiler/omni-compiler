subroutine sub(a)
    common /cmn/ n
    integer a(n, n)
    !$omp parallel private(a) shared(n)
    a(1, 1) = 1
    print *,a
    !$omp end parallel
end subroutine

