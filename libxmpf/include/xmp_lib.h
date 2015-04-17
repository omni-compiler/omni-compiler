!-------------------------------
!  descriptor
!-------------------------------
      type xmp_desc
         sequence
         integer*8 :: desc
      end type xmp_desc

!-------------------------------
!  coarray intrinsics
!-------------------------------
!! inquiry functions
!     integer, external :: image_index
!     integer, external :: lcobound, ucobound

!! transformation functions
      integer, external :: num_images, this_image

!-------------------------------
!  coarray runtime interface
!-------------------------------
!! synchronization
      include "xmp_lib_coarray_sync.h"

!! reference of coindexed objects
      include "xmp_lib_coarray_get.h"

!! allocate statement
      include "xmp_lib_coarray_alloc.h"

!! others
      integer, external :: xmpf_coarray_image

!      external :: xmpf_coarray_proc_init
!      external :: xmpf_coarray_proc_finalize

!      interface
!         subroutine xmpf_coarray_proc_init(tag)
!           integer(8), intent(out):: tag
!         end subroutine xmpf_coarray_proc_init
!         subroutine xmpf_coarray_proc_finalize(tag)
!           integer(8), intent(in):: tag
!         end subroutine xmpf_coarray_proc_finalize
!      end interface

!! hidden utilities
      integer, external :: xmpf_coarray_allocated_bytes
      integer, external :: xmpf_coarray_garbage_bytes

