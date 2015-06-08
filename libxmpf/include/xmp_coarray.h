!-------------------------------
!  coarray intrinsics
!-------------------------------
!! inquiry functions
!     integer, external :: image_index
!     integer, external :: lcobound, ucobound

!! transformation functions
      integer, external :: num_images, this_image

      interface xmpf_this_image
         function xmpf_this_image_coarray_wrap(descptr, corank) result(image)
           integer(8), intent(in) :: descptr
           integer, intent(in) :: corank
           integer image(corank)
         end function xmpf_this_image_coarray_wrap
         function xmpf_this_image_coarray_dim(descptr, corank, dim) result(coindex)
           integer(8), intent(in) :: descptr
           integer, intent(in) :: corank, dim
           integer coindex
         end function xmpf_this_image_coarray_dim
      end interface


!-------------------------------
!  coarray runtime interface
!-------------------------------
!! synchronization
      include "xmp_coarray_sync.h"

!! reference of coindexed objects
      include "xmp_coarray_get.h"

!! allocate statement
      include "xmp_coarray_alloc.h"

!! hidden utilities
      integer, external :: xmpf_coarray_allocated_bytes
      integer, external :: xmpf_coarray_garbage_bytes

