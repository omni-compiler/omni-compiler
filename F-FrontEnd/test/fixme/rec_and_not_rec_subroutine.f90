recursive subroutine s(a)
    if (a > 0) then
        call s(a - 1)
    end if
end subroutine s

! check subroutine t has no recursive attribute.
subroutine t(a)
    if (a > 0) then
        call s(a - 1)
    end if
end subroutine t
