module mod
    contains
        subroutine s
            integer a
            common /cmn/ a
            !$omp threadprivate(/cmn/)
            a = 1
        end subroutine
end module

