subroutine sub(a)
    common /cmn1/ c1
    common /cmn2/ c2
    integer a(c1, c1)
    integer i
    integer c1, c2
    !$omp parallel shared(a) private(i)
    a(1, 1) = 1
    c2 = 1
    i = 1
    print *,a
    !$omp end parallel
end subroutine

