subroutine sub
    integer a
    !$omp parallel sections
    a = 11
    a = 12
    !$omp section
    a = 21
    a = 22
    !$omp section
    a = 23
    a = 24
    !$omp end parallel sections 
end subroutine
