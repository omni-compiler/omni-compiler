!-------------------------------
!  descriptor
!-------------------------------
      type xmp_desc
         sequence
         integer*8 :: desc
      end type xmp_desc

!-------------------------------
!  coarray 
!-------------------------------
      include "xmp_lib_coarray_sync.h"

!!      integer, external :: image_index
!!      integer, external :: lcobound, ucobound
      integer, external :: num_images, this_image

!-------------------------------
!  array functions to support reference of coindexed-objects
!-------------------------------
      include "xmp_lib_coarray_get.h"

