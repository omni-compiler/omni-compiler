program test
    type tt
        integer a, b
        character(LEN=20) s
        integer c(10)
    end type

    type(tt)::t
    type(tt)::u
    t = tt(1, 2, "Happy", 3)
    u = t
!    t = tt(1, 2)
!    t = tt(1, "a", "Happy")
!    t = tt(1, "Happy")
    print *, t
    print *, u
end
