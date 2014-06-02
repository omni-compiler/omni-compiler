subroutine sub(n)
    integer a(0:n, 0:n)
    !$omp parallel private(a)
    a = 1
    print *,a
    !$omp end parallel
end subroutine

