!! This file is alive only for compatibility.

!! These functions are not solved yet.
      integer, external :: num_images, this_image   !! raw name libraries

!! inquiry functions
      integer, external :: xmpf_image_index
!     interface
!        integer function xmpf_image_index(descptr, coindexes)
!          integer(8), intent(in) :: descptr
!          integer, intent(in) :: coindexes(*)
!        end function xmpf_image_index_coarray_sub
!     end interface

!! hidden utilities
      integer, external :: xmpf_coarray_allocated_bytes
      integer, external :: xmpf_coarray_garbage_bytes

