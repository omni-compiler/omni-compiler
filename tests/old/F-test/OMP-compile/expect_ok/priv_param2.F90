subroutine sub2(n, a)
    integer :: a(n)
    a(1) = 1
end subroutine

subroutine sub1
    integer,parameter :: n = 3
    integer :: a(n)
    !$omp parallel default(private) shared(a)
    call sub(n, a)
    !$omp end parallel
end subroutine
