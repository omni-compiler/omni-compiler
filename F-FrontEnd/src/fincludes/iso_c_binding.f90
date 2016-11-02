module iso_c_binding
    implicit none

    ! TODO detect architecture and set correct value
    integer, parameter :: POINTER_LEN = 4 ! 32 bit
    ! integer, parameter :: POINTER_LEN = 8 ! 64 bit

    integer (KIND=4), parameter :: C_INT = 4
    integer (KIND=4), parameter :: C_SHORT = 2
    integer (KIND=4), parameter :: C_LONG = POINTER_LEN
    integer (KIND=4), parameter :: C_LONG_LONG = 8
    integer (KIND=4), parameter :: C_SIGNED_CHAR = 1
    integer (KIND=4), parameter :: C_SIZE_T = POINTER_LEN

    integer (KIND=4), parameter :: C_INT8_T = 1
    integer (KIND=4), parameter :: C_INT16_T = 2
    integer (KIND=4), parameter :: C_INT32_T = 4
    integer (KIND=4), parameter :: C_INT64_T = 8
    integer (KIND=4), parameter :: C_INT_LEAST8_T = 1
    integer (KIND=4), parameter :: C_INT_LEAST16_T = 2
    integer (KIND=4), parameter :: C_INT_LEAST32_T = 4
    integer (KIND=4), parameter :: C_INT_LEAST64_T = 8
    integer (KIND=4), parameter :: C_INT_FAST8_T = 1
    integer (KIND=4), parameter :: C_INT_FAST16_T = POINTER_LEN
    integer (KIND=4), parameter :: C_INT_FAST32_T = POINTER_LEN
    integer (KIND=4), parameter :: C_INT_FAST64_T = 8
    integer (KIND=4), parameter :: C_INTMAX_T = 8
    integer (KIND=4), parameter :: C_INTPTR_T = POINTER_LEN

    integer (KIND=4), parameter :: C_FLOAT = 4
    integer (KIND=4), parameter :: C_DOUBLE = 8
    integer (KIND=4), parameter :: C_LONG_DOUBLE = C_LONG *2

    integer (KIND=4), parameter :: C_FLOAT_COMPLEX = C_FLOAT
    integer (KIND=4), parameter :: C_DOUBLE_COMPLEX = C_DOUBLE
    integer (KIND=4), parameter :: C_LONG_DOUBLE_COMPLEX = C_LONG_DOUBLE

    integer (KIND=4), parameter :: C_BOOL = 1

    integer (KIND=4), parameter :: C_CHAR = 1
      
    character (KIND=1, LEN=1), parameter :: C_NULL_CHAR = achar(0)
    character (KIND=1, LEN=1), parameter :: C_ALERT = achar(7)
    character (KIND=1, LEN=1), parameter :: C_BACKSPACE = achar(8)
    character (KIND=1, LEN=1), parameter :: C_FORM_FEED = achar(12)
    character (KIND=1, LEN=1), parameter :: C_NEW_LINE = achar(10)
    character (KIND=1, LEN=1), parameter :: C_CARRIAGE_RETURN = achar(13)
    character (KIND=1, LEN=1), parameter :: C_HORIZONTAL_TAB = achar(9)
    character (KIND=1, LEN=1), parameter :: C_VERTICAL_TAB = achar(11)
 
    type, BIND(C) :: C_PTR
        private
        integer(C_INTPTR_T) :: ptr
    end type C_PTR

    type, BIND(C) :: C_FUNPTR
        private
        integer(C_INTPTR_T) :: ptr
    end type C_FUNPTR

    type(C_PTR), parameter :: C_NULL_PTR = C_PTR(0)
    type(C_FUNPTR), parameter :: C_NULL_FUNPTR = C_FUNPTR(0)

    integer(4), parameter, private :: for_desc_max_rank = 7    
    integer(C_INTPTR_T), parameter, private :: for_desc_array_defined= 1
    integer(C_INTPTR_T), parameter, private :: for_desc_array_nodealloc = 2
    integer(C_INTPTR_T), parameter, private :: for_desc_array_contiguous = 4
    integer(C_INTPTR_T), parameter, private :: for_desc_flags = & 
                                               for_desc_array_defined + &
                                               for_desc_array_nodealloc + &
                                               for_desc_array_contiguous

    type, private :: for_desc_triplet
        integer(C_INTPTR_T) :: extent
        integer(C_INTPTR_T) :: mult  ! multiplier for this dimension
        integer(C_INTPTR_T) :: lowerbound
    end type for_desc_triplet

    type, private :: for_array_descriptor
        integer(C_INTPTR_T) :: base
        integer(C_INTPTR_T) :: len  ! len of data type
        integer(C_INTPTR_T) :: offset
        integer(C_INTPTR_T) :: flags
        integer(C_INTPTR_T) :: rank
        integer(C_INTPTR_T) :: reserved1
        type(for_desc_triplet) :: diminfo(for_desc_max_rank)
    end type for_array_descriptor

    interface c_associated
        module procedure c_associated_ptr, c_associated_funptr
    end interface

    interface c_f_pointer
        module procedure c_f_pointer_scalar

        subroutine c_f_pointer_array1 (cptr, fptr, shape)
            import :: c_ptr 
            implicit none
            type(c_ptr), intent(in) :: cptr
            integer, POINTER, intent(out) :: fptr(:)
            integer(1), intent(in) :: shape(:)
        end subroutine c_f_pointer_array1

        subroutine c_f_pointer_array2 (cptr, fptr, shape)
            import :: c_ptr
            implicit none
            type(c_ptr), intent(in) :: cptr
            integer, POINTER, intent(out) :: fptr(:)
            integer(2), intent(in) :: shape(:)
        end subroutine c_f_pointer_array2

        subroutine c_f_pointer_array4 (cptr, fptr, shape)
            import :: c_ptr
            implicit none
            type(c_ptr), intent(in) :: cptr
            integer, POINTER, intent(out) :: fptr(:)
            integer(4), intent(in) :: shape(:)
        end subroutine c_f_pointer_array4

        subroutine c_f_pointer_array8 (cptr, fptr, shape) 
            import :: c_ptr
            implicit none
            type(c_ptr), intent(in) :: cptr
            integer, POINTER, intent(out) :: fptr(:)
            integer(8), intent(in) :: shape(:)
        end subroutine c_f_pointer_array8

    end interface

    private :: c_f_pointer_private1
    private :: c_f_pointer_private2
    private :: c_f_pointer_private4
    private :: c_f_pointer_private8

CONTAINS

    function c_associated_ptr (c_ptr_1, c_ptr_2)
        logical(4) :: c_associated_ptr
        type(c_ptr) :: c_ptr_1
        type(c_ptr), optional :: c_ptr_2
    end function c_associated_ptr

    function c_associated_funptr (c_ptr_1, c_ptr_2)
        logical(4) :: c_associated_funptr
        type(c_funptr) :: c_ptr_1
        type(c_funptr), optional :: c_ptr_2
    end function c_associated_funptr

    subroutine c_f_pointer_scalar (cptr, fptr)
        integer, POINTER , intent(in):: cptr 
        integer, POINTER , intent(out):: fptr
    end subroutine c_f_pointer_scalar

    subroutine c_f_pointer_private1 (caddr, fdesc, shape)
        integer(C_INTPTR_T), intent(in) :: caddr
        type(for_array_descriptor), intent(inout) :: fdesc
        integer(1), intent(in) :: shape(:)
    end subroutine c_f_pointer_private1

    subroutine c_f_pointer_private2 (caddr, fdesc, shape)
        integer(C_INTPTR_T), intent(in) :: caddr
        type(for_array_descriptor), intent(inout) :: fdesc
        integer(2), intent(in) :: shape(:)
    end subroutine c_f_pointer_private2

    subroutine c_f_pointer_private4 (caddr, fdesc, shape)
        integer(C_INTPTR_T), intent(in) :: caddr
        type(for_array_descriptor), intent(inout) :: fdesc
        integer(4), intent(in) :: shape(:)
    end subroutine c_f_pointer_private4

    subroutine c_f_pointer_private8 (caddr, fdesc, shape)
        integer(C_INTPTR_T), intent(in) :: caddr
        type(for_array_descriptor), intent(inout) :: fdesc
        integer(8), intent(in) :: shape(:)
    end subroutine c_f_pointer_private8

    subroutine c_f_procpointer (cptr, fptr)
        integer, POINTER , intent(in):: cptr
        ! TODO remove comment when supported in OMNI
        !procedure(), POINTER , intent(out):: fptr   
        integer, POINTER , intent(out):: fptr
    end subroutine c_f_procpointer

    function c_funloc (x)
        type(c_funptr) :: c_funloc
        integer :: x
    end function c_funloc

    function c_loc (x)
        type(c_ptr) :: c_loc
        integer :: x
    end function c_loc

end module iso_c_binding
