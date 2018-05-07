module generic_mod
    implicit none
    private

    type, public :: vector_ref
        private
        integer :: current_size = 0
        integer :: maximum_size = 0    
    contains
        procedure :: add_item
        generic :: add => add_item
    end type vector_ref
    
contains       

    subroutine add_item(this)
        class(vector_ref), intent(inout) :: this
    end subroutine add_item

end module generic_mod
