subroutine sub1(a, b, c)
    integer,intent(in)::a
    integer,intent(out)::b
    integer,intent(inout)::c
end subroutine

subroutine sub2(a, b, c)
    intent(in)::a
    intent(out)::b
    intent(inout)::c
end subroutine
