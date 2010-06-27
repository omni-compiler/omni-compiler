program main
    integer a(0:1, 2:3, 4:5)
    integer i
    common /cmn/ a
    !$omp threadprivate (/cmn/)

    !$omp parallel private(i)
    a = 1
    do i = 1, 3
10      format("dim"I1": lb="I2" ub="I2)    
        write(*,10) i, lbound(a, i), ubound(a, i)
    end do

    !$omp end parallel
end program

