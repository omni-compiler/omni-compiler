subroutine sub1()
    integer,save::a
end subroutine

subroutine sub2()
    save::a
end subroutine

subroutine sub3()
    integer a
    save
end subroutine

subroutine sub4()
    integer a
    save :: a
end subroutine
