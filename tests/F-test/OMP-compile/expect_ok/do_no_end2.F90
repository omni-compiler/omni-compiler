subroutine sub
    integer i
    !$omp do
    do i = 1, 3
        print *,i
    end do
end subroutine

