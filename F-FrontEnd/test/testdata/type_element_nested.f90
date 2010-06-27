program end
    type st1
        integer a
    end type

    type st2
        type(st1) t1
    end type

    type st3
        type(st2) t2(2)
    end type

    type(st3)::t3
    t3%t2(1)%t1%a = 1
end

