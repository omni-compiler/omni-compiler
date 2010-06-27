subroutine sub(a)
    integer::a(*)
    !$omp parallel
    a(1) = 1
    !$omp end parallel
end subroutine
