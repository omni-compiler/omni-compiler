subroutine sub
    integer a
    a = 0
    !$omp parallel &
    !$omp default(private) reduction(+:a)
    a = a + 1
    !$omp end parallel
end subroutine

