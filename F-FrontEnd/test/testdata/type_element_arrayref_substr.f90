    program main

        character(len=5) a

        type st
            character(len=5)::n(10)
        end type

        type su
            type(st)::t
        end type

        type(su)::u

        a = u%t%n(8)(1:5)

    end program

