subroutine sub(i)
    integer i
    integer, external::omp_get_thread_num
    !$omp ordered
    print *, "th=", omp_get_thread_num(), "i=", i
    !$omp end ordered
end subroutine

program main
    integer i
    !$omp parallel do ordered
    do i = 1, 3
        call sub(i)
    end do
    !$omp end parallel do
end program

