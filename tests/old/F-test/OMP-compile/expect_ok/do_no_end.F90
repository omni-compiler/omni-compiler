program main
    integer i
    !$omp parallel
    !$omp do
    do i = 1, 3
        print *,i
    end do
    !$omp end parallel
end

