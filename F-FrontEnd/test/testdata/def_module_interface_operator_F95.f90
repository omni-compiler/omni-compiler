module m1
    type st
        integer v
    end type

    interface operator(==)
        module procedure binop
    end interface

    interface operator(/=)
        module procedure binop
    end interface

    interface operator(<)
        module procedure binop
    end interface

    interface operator(>)
        module procedure binop
    end interface

    interface operator(>=)
        module procedure binop
    end interface

    interface operator(<=)
        module procedure binop
    end interface

    contains
        function binop(x, y)
            implicit none
            type(st)::binop, x, y
            intent(in)::x, y
            binop%v = x%v + y%v
        end function
end module

