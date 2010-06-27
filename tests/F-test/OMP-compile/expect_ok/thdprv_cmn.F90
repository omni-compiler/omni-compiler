program main
    integer a(0:1)
    common /cmn/ a
    !$omp threadprivate (/cmn/)

    !$omp parallel
    a = 1
    print *,"lb=",lbound(a),",ub=",ubound(a)
    !$omp end parallel
end program

