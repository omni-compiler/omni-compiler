subroutine s
    integer,save::b(3, 2)
    !$omp threadprivate(b)
    !$omp parallel
    b(3, 2) = 9
    !$omp end parallel
end subroutine
