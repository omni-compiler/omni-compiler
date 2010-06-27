        subroutine sub()
            integer a(:), b
            allocatable a, b(:)
            allocate(a(2), b(3))
        end subroutine

