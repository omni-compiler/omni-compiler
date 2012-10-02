    program main

        type TT
	    integer, pointer :: a
        end type TT
	type(TT), pointer :: b
	type(TT), pointer :: c(:)

	b => null()
	c => null()

    end program main
