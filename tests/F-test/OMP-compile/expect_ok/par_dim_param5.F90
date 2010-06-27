subroutine sub(a)
    integer n
    integer a(n, n)
    common /cmn/ n
    !$omp parallel default(shared) private(a)
    a(1, 1) = 1
    print *,a
    !$omp end parallel
end subroutine

