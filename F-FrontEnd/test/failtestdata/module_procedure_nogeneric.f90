module m
    interface
        ! expected error of
        ! MODULE_PROCEDURE must be in a generic module interface
        module procedure s1
    end interface
    contains
        subroutine s1
        end subroutine
end module

