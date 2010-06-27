module par_dim_mod_m
    integer,parameter::n = kind(1)
end module

subroutine s1
    use par_dim_mod_m
    integer(kind=n)::t(2)
    common /cmn/ t
!$omp threadprivate(/cmn/)
    t = 1
end subroutine

subroutine s2
    use par_dim_mod_m
    integer(kind=n)::a
    integer(kind=n)::t(2)
    common /cmn/ t
!$omp threadprivate(/cmn/)
!$omp parallel
    a = 2
    t = 2
!$omp end parallel
end subroutine
