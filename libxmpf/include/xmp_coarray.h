!-------------------------------
!  coarray intrinsics
!-------------------------------
!! inquiry functions
      integer, external :: image_index
      integer, external :: xmpf_image_index
!     interface
!        integer function xmpf_image_index(descptr, coindexes)
!          integer(8), intent(in) :: descptr
!          integer, intent(in) :: coindexes(*)
!        end function xmpf_image_index_coarray_sub
!     end interface

      integer, external :: lcobound, ucobound
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
      integer, external :: num_images, this_image
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

!! reference of coindexed objects
      include "xmp_coarray_get.h"

!! allocate statement
      include "xmp_coarray_alloc.h"

!! reduction subroutines
!!      include "xmp_coarray_reduction.h"
      interface co_sum
         subroutine co_sum_i4(source, result)
           integer(4), intent(in)  :: source
           integer(4), intent(out) :: result
         end subroutine co_sum_i4
         subroutine co_sum_r4(source, result)
           real(4), intent(in)  :: source
           real(4), intent(out) :: result
         end subroutine co_sum_r4
         subroutine co_sum_r8(source, result)
           real(8), intent(in)  :: source
           real(8), intent(out) :: result
         end subroutine co_sum_r8
      end interface

      interface co_max
         subroutine co_max_i4(source, result)
           integer(4), intent(in)  :: source
           integer(4), intent(out) :: result
         end subroutine co_max_i4
         subroutine co_max_r4(source, result)
           real(4), intent(in)  :: source
           real(4), intent(out) :: result
         end subroutine co_max_r4
         subroutine co_max_r8(source, result)
           real(8), intent(in)  :: source
           real(8), intent(out) :: result
         end subroutine co_max_r8
      end interface

      interface co_min
         subroutine co_min_i4(source, result)
           integer(4), intent(in)  :: source
           integer(4), intent(out) :: result
         end subroutine co_min_i4
         subroutine co_min_r4(source, result)
           real(4), intent(in)  :: source
           real(4), intent(out) :: result
         end subroutine co_min_r4
         subroutine co_min_r8(source, result)
           real(8), intent(in)  :: source
           real(8), intent(out) :: result
         end subroutine co_min_r8
      end interface

!! hidden utilities
      integer, external :: xmpf_coarray_allocated_bytes
      integer, external :: xmpf_coarray_garbage_bytes

