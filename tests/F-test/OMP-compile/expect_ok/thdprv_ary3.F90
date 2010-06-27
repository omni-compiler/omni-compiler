subroutine sub
    integer,save::b(1024 * 1024 * 10, 1)
    !$omp threadprivate(b)
    !$omp parallel
    b(3, 2) = 9
    !$omp end parallel
end subroutine

program main
    external sub
    call sub()
end program
