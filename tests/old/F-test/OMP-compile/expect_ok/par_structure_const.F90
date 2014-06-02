module structure_ary_m
    type tt
        integer a
    end type
end module

subroutine s
    use structure_ary_m
    type(tt),save::t(5)
    !$omp parallel
    t(1) = tt(1)
    !$omp end parallel
end subroutine

