subroutine s
    integer a
    !$omp parallel if(.TRUE.)
    a = 1
    !$omp end parallel
end subroutine
