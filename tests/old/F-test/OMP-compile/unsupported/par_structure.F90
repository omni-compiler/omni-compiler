program main
    type tt
        integer tta
    end type
    integer, external::omp_get_thread_num
    type(tt)::t
    !$omp parallel
    t%tta = omp_get_thread_num()
    !$omp end parallel
end program 

