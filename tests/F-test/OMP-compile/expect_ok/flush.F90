program main
    integer a
    common /cmn/ a
    integer,save::b
    !$omp flush(/cmn/, b)
    print *, "after flush"
end program
