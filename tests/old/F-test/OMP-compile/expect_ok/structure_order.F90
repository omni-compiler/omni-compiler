module structure_order_m
    type :: ta
        integer a
    end type
    type :: tb
        type(ta)::b
    end type
    type :: tc
        type(tb)::c
    end type
    type(tc)::d
end module

