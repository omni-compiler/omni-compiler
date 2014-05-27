program main
    integer i, a
    a = 0
    !$omp parallel do lastprivate(a)
    do i = 1, 3
        a = a + 1
    end do
    !$omp end parallel do
end program

