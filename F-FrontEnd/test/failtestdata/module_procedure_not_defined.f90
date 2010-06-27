module m
    interface s
        ! expected error of
        ! "s2 is not defined in module"
        module procedure s1, s2
    end interface
    contains
        subroutine s1()
        end subroutine
end module

