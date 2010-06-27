subroutine s
    integer,save::a, b(3)
    !$omp threadprivate(a, b)
    !$omp parallel
    a = 1
    b(2) = 9
    !$omp end parallel
end subroutine
