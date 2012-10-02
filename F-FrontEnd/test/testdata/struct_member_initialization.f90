    program main

        type TT0
            integer :: a = 10
        end type TT0
        type TT1
	    type(TT0), pointer :: c(:) => null()
        end type TT1

	type(TT1), pointer :: p(:) = null()

    end program main
