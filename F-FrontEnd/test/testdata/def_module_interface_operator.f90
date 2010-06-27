module m
    type t
        integer n
    end type

    interface operator(+)
        module procedure bin_ope
    end interface

    interface operator(-)
        module procedure bin_ope
    end interface

    interface operator(/)
        module procedure bin_ope
    end interface

    interface operator(*)
        module procedure bin_ope
    end interface

    interface operator(==)
        module procedure cmp_ope
    end interface

    interface operator(>=)
        module procedure cmp_ope
    end interface

    interface operator(<=)
        module procedure cmp_ope
    end interface

    interface operator(/=)
        module procedure cmp_ope
    end interface

    interface assignment(=)
        module procedure asg_ope
    end interface

    contains
        integer function bin_ope(a, b)
            type(t),intent(in)::a, b
            bin_ope = 1
        end function

        integer function cmp_ope(a, b)
            type(t),intent(in)::a, b
            cmp_ope = 1
        end function

        subroutine asg_ope(a, b)
            type(t),intent(inout)::a
            type(t),intent(in)::b
        end subroutine
end module
