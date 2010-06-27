function func(a, b, c)
    integer a
    intent(in) a
    integer b
    intent(inout) b, c
    integer c
    func = 1
end function

subroutine sub(a, b, c)
    integer a
    intent(in)::a
    intent(out)::b
    integer b
    intent(inout)::c
    integer c
end subroutine
