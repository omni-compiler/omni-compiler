subroutine sub(a)
    common /cmn/ n
    integer a(n, n)
    !$omp parallel shared(a, n)
    a(1, 1) = 1
    print *,a
    !$omp end parallel
end subroutine

