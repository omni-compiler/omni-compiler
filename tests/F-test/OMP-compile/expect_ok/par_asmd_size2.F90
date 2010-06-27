subroutine s(x, a)
    integer x
    integer a(x, *)
    !$omp parallel shared(a)
    a(1, 1) = 1
    !$omp end parallel
end subroutine

