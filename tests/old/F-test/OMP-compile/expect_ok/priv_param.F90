subroutine sub
    integer,parameter :: n = 3
    integer :: a(n)
    !$omp parallel default(private) shared(a)
    a(1) = 1
    !$omp end parallel
end subroutine
