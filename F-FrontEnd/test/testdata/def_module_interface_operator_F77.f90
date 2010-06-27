module m1
    type st
        integer v
    end type

    interface assignment(=)
        module procedure asgop
    end interface

    interface operator(+)
        module procedure binop
    end interface

    interface operator(-)
        module procedure binop
    end interface

    interface operator(*)
        module procedure binop
    end interface

    interface operator(/)
        module procedure binop
    end interface

    interface operator(**)
        module procedure binop
    end interface

    interface operator(//)
        module procedure binop
    end interface

    interface operator(.eq.)
        module procedure binop
    end interface

    interface operator(.ne.)
        module procedure binop
    end interface

    interface operator(.lt.)
        module procedure binop
    end interface

    interface operator(.gt.)
        module procedure binop
    end interface

    interface operator(.le.)
        module procedure binop
    end interface

    interface operator(.ge.)
        module procedure binop
    end interface

    interface operator(.and.)
        module procedure binop
    end interface

    interface operator(.or.)
        module procedure binop
    end interface

    interface operator(.eqv.)
        module procedure binop
    end interface

    interface operator(.neqv.)
        module procedure binop
    end interface

    interface operator(.not.)
        module procedure unaop
    end interface

    contains
        subroutine asgop(x, y)
            implicit none
            type(st)::x, y
            intent(out)::x
            intent(in)::y
            x%v = y%v
        end subroutine

        function binop(x, y)
            implicit none
            type(st)::binop, x, y
            intent(in)::x, y
            binop%v = x%v + y%v
        end function

        function unaop(x)
            implicit none
            type(st)::unaop, x
            intent(in)::x
            unaop%v = x%v
        end function
end module

