program main
    type tt
        integer,pointer::a
    end type

    integer n, s
    integer,pointer::a0
    integer,allocatable,dimension(:)::a1
    real,pointer,dimension(:)::a2
    real,allocatable,dimension(:, :)::a3
    character,pointer::ch
    type(tt) t

    deallocate(a0)
    deallocate(a1)
    deallocate(a2, a3)
    deallocate(ch)
    deallocate(a0, stat=s)
    deallocate(a1, stat=s)
    deallocate(a2, a3, stat=s)
    deallocate(t%a)
end

