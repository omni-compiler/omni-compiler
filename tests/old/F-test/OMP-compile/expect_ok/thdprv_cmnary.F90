program main
    integer,parameter::nn=3
    integer,parameter::n=nn * 3
    integer a(n)
    common /cmn/ a
    !$omp threadprivate (/cmn/)

    !$omp parallel
    a = 1
    !$omp end parallel
end program

