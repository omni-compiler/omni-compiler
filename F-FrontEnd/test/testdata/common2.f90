subroutine sub1(x)
    integer x
end subroutine

subroutine sub2()
    integer x(10)
    common /cmn/ x
    call sub1(x(1))
end subroutine

