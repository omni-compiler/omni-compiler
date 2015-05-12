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
      include "xmp_coarray_sync.h"

!! reference of coindexed objects
      include "xmp_coarray_get.h"

!! allocate statement
      include "xmp_coarray_alloc.h"

!! hidden utilities
      integer, external :: xmpf_coarray_allocated_bytes
      integer, external :: xmpf_coarray_garbage_bytes

