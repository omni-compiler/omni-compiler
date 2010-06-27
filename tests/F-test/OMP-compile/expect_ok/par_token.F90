subroutine s
    integer a, b
    !$omp parallel private(a) , shared(b)
    a = b
    !$omp end parallel
end subroutine
