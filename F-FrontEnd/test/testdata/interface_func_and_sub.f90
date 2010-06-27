! function and subroutine in interface
program main
    interface
        character function f()
        end function
        subroutine s(i)
            real i
        end subroutine
    end interface

    character a

    a = f()
    call s(1.0)
end
