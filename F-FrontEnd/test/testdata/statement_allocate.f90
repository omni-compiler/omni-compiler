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

    allocate(a0)
    allocate(a1(1))
    allocate(a2(1:2), a3(3, 1:n))
    allocate(ch)
    allocate(a0, stat=s)
    allocate(a1(1), stat=s)
    allocate(a2(1:2), a3(3, 1:n), stat=s)
    allocate(t%a)
end

