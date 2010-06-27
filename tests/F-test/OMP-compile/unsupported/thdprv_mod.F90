module thdprv_m
    integer::a
    !$omp threadprivate(a)
end module

subroutine s
    use thdprv_m
    a = 1
end subroutine

program main
    !$omp parallel
    call s()
    !$omp end parallel
end program

