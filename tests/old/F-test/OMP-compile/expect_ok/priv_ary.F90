subroutine s
    integer,parameter :: m = 3
    integer,parameter :: n = m
    integer p(n)
    !$omp parallel private(p)
    p = 1
    !$omp end parallel
end subroutine
