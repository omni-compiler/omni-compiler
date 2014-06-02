module structure_ary_m
    type tt
        integer a
    end type
end module

subroutine s
    use structure_ary_m
    type(tt)::t(5)
    !$omp parallel
    t(1)%a = 1
    !$omp end parallel
end subroutine

