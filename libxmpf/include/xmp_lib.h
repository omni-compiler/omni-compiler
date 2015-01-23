!-------------------------------
!  descriptor
!-------------------------------
      type xmp_desc
         sequence
         integer*8 :: desc
      end type xmp_desc

!-------------------------------
!  coarray statements
!-------------------------------
      interface
         subroutine xmp_sync_all(stat, errmsg)
         integer, optional, intent(out) :: stat
         character(len=*), optional, intent(out) :: errmsg
         end subroutine
      end interface

      interface
        subroutine xmp_sync_memory(stat, errmsg)
         integer, optional, intent(out) :: stat
         character(len=*), optional, intent(out) :: errmsg
         end subroutine
      end interface

      interface xmp_sync_images
         subroutine xmp_sync_images_one(image, stat, errmsg)
         integer, intent(in) :: image
         integer, optional, intent(out) :: stat
         character(len=*), optional, intent(out) :: errmsg
         end subroutine
         subroutine xmp_sync_images_ast(image, stat, errmsg)
         character(len=1), intent(in) :: image
         integer, optional, intent(out) :: stat
         character(len=*), optional, intent(out) :: errmsg
         end subroutine
      end interface

      interface
         subroutine xmp_lock(stat, errmsg)
         integer, optional, intent(out) :: stat
         character(len=*), optional, intent(out) :: errmsg
         end subroutine
      end interface

      interface
         subroutine xmp_unlock(stat, errmsg)
         integer, optional, intent(out) :: stat
         character(len=*), optional, intent(out) :: errmsg
         end subroutine
      end interface

      interface
         subroutine xmp_critical
         end subroutine
      end interface

      interface
         subroutine xmp_end_critical
         end subroutine
      end interface

      interface
         subroutine xmp_error_stop
         end subroutine
      end interface

!-------------------------------
!  coarray intrinsic functions
!-------------------------------
!!      integer, external :: image_index
!!      integer, external :: lcobound, ucobound
      integer, external :: num_images, this_image

!-------------------------------
!  coarray atomic subroutines
!-------------------------------
! Exactly, atom should be integer(kind=atomc_int_kind) or 
! logical(kind=atomic_logical_kind), whose kind is defined in
! the intrinsic module iso_fortran_env [J.Reid, N1824:15.3].

      interface atmic_define
         subroutine atomic_define_i2(atom, value)
         integer, intent(out) :: atom
         integer(2), intent(in) :: value
         end subroutine
         subroutine atomic_define_i4(atom, value)
         integer, intent(out) :: atom
         integer(4), intent(in) :: value
         end subroutine
         subroutine atomic_define_i8(atom, value)
         integer, intent(out) :: atom
         integer(8), intent(in) :: value
         end subroutine
         subroutine atomic_define_l2(atom, value)
         logical, intent(out) :: atom
         logical(2), intent(in) :: value
         end subroutine
         subroutine atomic_define_l4(atom, value)
         logical, intent(out) :: atom
         logical(4), intent(in) :: value
         end subroutine
         subroutine atomic_define_l8(atom, value)
         logical, intent(out) :: atom
         logical(8), intent(in) :: value
         end subroutine
      end interface

      interface atmic_ref
         subroutine atomic_ref_i2(value, atom)
         integer(2), intent(out) :: value
         integer, intent(in) :: atom
         end subroutine
         subroutine atomic_ref_i4(value, atom)
         integer(4), intent(out) :: value
         integer, intent(in) :: atom
         end subroutine
         subroutine atomic_ref_i8(value, atom)
         integer(8), intent(out) :: value
         integer, intent(in) :: atom
         end subroutine
         subroutine atomic_ref_l2(value, atom)
         logical(2), intent(out) :: value
         logical, intent(in) :: atom
         end subroutine
         subroutine atomic_ref_l4(value, atom)
         logical(4), intent(out) :: value
         logical, intent(in) :: atom
         end subroutine
         subroutine atomic_ref_l8(value, atom)
         logical(8), intent(out) :: value
         logical, intent(in) :: atom
         end subroutine
      end interface

      include "xmp_lib_coarray_get.h"


