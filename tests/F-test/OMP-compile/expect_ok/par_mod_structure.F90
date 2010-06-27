module par_mod
    type tt
        integer tta
    end type
end module

program main
    use par_mod
    integer, external::omp_get_thread_num
    type(tt)::t
    !$omp parallel
    t%tta = omp_get_thread_num()
    !$omp end parallel
end program 

