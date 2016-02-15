!-------------------------------
!  coarray intrinsics
!-------------------------------
!! inquiry functions
      integer, external :: xmpf_image_index
!     interface
!        integer function xmpf_image_index(descptr, coindexes)
!          integer(8), intent(in) :: descptr
!          integer, intent(in) :: coindexes(*)
!        end function xmpf_image_index_coarray_sub
!     end interface

      interface xmpf_cobound
         !! restriction: kind must be 4.
         function xmpf_cobound_nodim(descptr, kind, lu, corank)         &
     &    result(bounds)
           integer(8), intent(in) :: descptr
           integer, intent(in) :: corank, lu, kind
           integer bounds(corank)           !! allocate here in Fortran
         end function xmpf_cobound_nodim
         !! restriction: kind must be 4.
         function xmpf_cobound_dim(descptr, dim, kind, lu, corank)      &
     &    result(bound)
           integer(8), intent(in) :: descptr
           integer, intent(in) :: corank, dim, lu, kind
           integer bound
         end function xmpf_cobound_dim
      end interface

!! transformation functions
      integer, external :: num_images, this_image   !! raw name libraries
      interface xmpf_this_image
         function xmpf_this_image_coarray_wrap(descptr, corank)         &
     &    result(image)
           integer(8), intent(in) :: descptr
           integer, intent(in) :: corank
           integer image(corank)
         end function xmpf_this_image_coarray_wrap
         function xmpf_this_image_coarray_dim(descptr, corank, dim)     &
     &    result(coindex)
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
!!!      include "xmp_coarray_sync_sxace.h"

!! reference of coindexed objects
      include "xmp_coarray_get.h"

!! allocate statement
      include "xmp_coarray_alloc.h"

!! reduction subroutines
      include "xmp_coarray_reduction.h"

!! hidden utilities
      integer, external :: xmpf_coarray_allocated_bytes
      integer, external :: xmpf_coarray_garbage_bytes

