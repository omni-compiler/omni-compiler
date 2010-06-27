subroutine sub2(n, a)
    integer :: a(n)
    !$omp parallel default(private) shared(a)
    a(1) = 1
    n = 1
    !$omp end parallel
end subroutine

subroutine sub1
    integer,parameter :: n = 3
    integer :: a(n)
    call sub2(n, a)
end subroutine
