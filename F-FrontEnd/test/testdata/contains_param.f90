        module m
            private
            integer a
            parameter(a = 1)

            contains
                subroutine sub
                    call s(a)
                end subroutine
        end module

