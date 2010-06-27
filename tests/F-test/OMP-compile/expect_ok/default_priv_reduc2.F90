subroutine sub
    integer a, i
    a = 0
    !$omp parallel default(private)
    !$omp do reduction(+:a)
    do i=1, 3
        a = a + 1
    end do
    !$omp end parallel
end subroutine

