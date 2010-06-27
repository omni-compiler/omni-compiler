        program main

            logical,dimension(10)::a

            contains
                subroutine sub()
                    integer::i
                    i = 1
                    a(i) = .TRUE.
                end subroutine
        end

