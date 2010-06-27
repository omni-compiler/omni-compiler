subroutine sub2
    integer a
    common /cmn/ a
    !$omp threadprivate(a)
    a = 222
end subroutine

subroutine sub1
    integer a
    common /cmn/ a
    !$omp threadprivate(a)
    !$omp parallel
    a = 111
    call sub2()
    !$omp end parallel
end subroutine

