module m
    type t
        real, pointer, dimension(:,:) :: a
    end type

    real, pointer, dimension(:,:) :: a

contains
    subroutine f(pp)
        type (t), target  :: pp
        type (t), pointer :: p
        p => pp
        a => p%a
    end subroutine
end module m
