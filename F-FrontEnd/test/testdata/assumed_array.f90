subroutine sub1(a1, a2, a3, a4, n)
    integer                     ::n
    integer, dimension(*)       ::a1
    integer, dimension(0:*)     ::a2
    integer, dimension(1,*)     ::a3
    integer, dimension(1,0:*)   ::a4
    integer, dimension(n)       ::a5
end

subroutine sub2(b1, b2)
    integer,dimension(:)            ::b1
    integer,dimension(0:)           ::b2
    integer,pointer,dimension(:,:)  ::b3
    integer s
    s = size(b1)
    s = size(b2)
    s = size(b3)
end

