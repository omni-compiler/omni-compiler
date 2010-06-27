subroutine sub
!       This is intrinsic function call
        write(*,*) verify("abc", "a")
end subroutine
program main
        integer i, j, k
!       This should be treated as external verify() subroutine call, not
!       intrinsic
        call verify(i, j, k)

!       This function call should cause error
!       because argument type is incompatible
!        j = verify(i, j, k)
end program
