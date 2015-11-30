!-------------------------------
!  sync all
!-------------------------------
      interface
!!         subroutine xmpf_sync_all(arg1, ...)
!!           class(*) :: arg1
!!         end subroutine xmpf_sync_all

         subroutine xmpf_sync_all_stat(stat, errmsg)
           integer, intent(out) :: stat
           character(len=*), intent(out), optional :: errmsg
         end subroutine xmpf_sync_all_stat
      end interface

!-------------------------------
!  sync memory
!-------------------------------
      interface xmpf_sync_memory
         subroutine xmpf_sync_memory_nostat()
         end subroutine xmpf_sync_memory_nostat
         subroutine xmpf_sync_memory_stat_wrap(stat, errmsg)
           integer, intent(out) :: stat
           character(len=*), intent(out), optional :: errmsg
         end subroutine xmpf_sync_memory_stat_wrap
      end interface

!-------------------------------
!  sync images
!-------------------------------
      interface xmpf_sync_images
         subroutine xmpf_sync_image_nostat(image)  ! no wrapper
           integer, intent(in) :: image
         end subroutine xmpf_sync_image_nostat_wrap
         subroutine xmpf_sync_images_nostat_wrap(images)
           integer, intent(in) :: images(:)
         end subroutine xmpf_sync_images_nostat_wrap
         subroutine xmpf_sync_allimages_nostat_wrap(aster)
           character(len=1), intent(in) :: aster
         end subroutine xmpf_sync_allimages_nostat_wrap

         subroutine xmpf_sync_image_stat_wrap(image, stat, errmsg)
           integer, intent(in) :: image
           integer, intent(out) :: stat
           character(len=*), intent(out), optional :: errmsg
         end subroutine xmpf_sync_image_stat_wrap
         subroutine xmpf_sync_images_stat_wrap(images, stat, errmsg)
           integer, intent(in) :: images(:)
           integer, intent(out) :: stat
           character(len=*), intent(out), optional :: errmsg
         end subroutine xmpf_sync_images_stat_wrap
         subroutine xmpf_sync_allimages_stat_wrap(aster, stat, errmsg)
           character(len=1), intent(in) :: aster
           integer, intent(out) :: stat
           character(len=*), intent(out), optional :: errmsg
         end subroutine xmpf_sync_allimages_stat_wrap
      end interface

!-------------------------------
!  coarray lock/unlock
!-------------------------------
!!!! not supported yet
      interface
         subroutine xmpf_lock(stat, errmsg)
         integer, optional, intent(out) :: stat
         character(len=*), optional, intent(out) :: errmsg
         end subroutine
      end interface

!!!! not supported yet
      interface
         subroutine xmpf_unlock(stat, errmsg)
         integer, optional, intent(out) :: stat
         character(len=*), optional, intent(out) :: errmsg
         end subroutine
      end interface

!-------------------------------
!  coarray critical construct
!-------------------------------
      interface
         subroutine xmpf_critical
         end subroutine
      end interface

      interface
         subroutine xmpf_end_critical
         end subroutine
      end interface

!-------------------------------
!  error stop
!-------------------------------
      interface
         subroutine xmpf_error_stop
         end subroutine
      end interface

!-------------------------------
!  coarray atomic subroutines
!-------------------------------
! Exactly, variable atom should be integer(kind=atomc_int_kind) or 
! logical(kind=atomic_logical_kind), whose kind is defined in the 
! intrinsic module iso_fortran_env [J.Reid, N1824:15.3].

      interface atomic_define
!!         subroutine atomic_define_i2(atom, value)
!!         integer, intent(out) :: atom
!!         integer(2), intent(in) :: value
!!         end subroutine
         subroutine atomic_define_i4(atom, value)
         integer, intent(out) :: atom
         integer(4), intent(in) :: value
         end subroutine
         subroutine atomic_define_i8(atom, value)
         integer, intent(out) :: atom
         integer(8), intent(in) :: value
         end subroutine
!!         subroutine atomic_define_l2(atom, value)
!!         logical, intent(out) :: atom
!!         logical(2), intent(in) :: value
!!         end subroutine
         subroutine atomic_define_l4(atom, value)
         logical, intent(out) :: atom
         logical(4), intent(in) :: value
         end subroutine
         subroutine atomic_define_l8(atom, value)
         logical, intent(out) :: atom
         logical(8), intent(in) :: value
         end subroutine
      end interface

      interface atomic_ref
!!         subroutine atomic_ref_i2(value, atom)
!!         integer(2), intent(out) :: value
!!         integer, intent(in) :: atom
!!         end subroutine
         subroutine atomic_ref_i4(value, atom)
         integer(4), intent(out) :: value
         integer, intent(in) :: atom
         end subroutine
         subroutine atomic_ref_i8(value, atom)
         integer(8), intent(out) :: value
         integer, intent(in) :: atom
         end subroutine
!!         subroutine atomic_ref_l2(value, atom)
!!         logical(2), intent(out) :: value
!!         logical, intent(in) :: atom
!!         end subroutine
         subroutine atomic_ref_l4(value, atom)
         logical(4), intent(out) :: value
         logical, intent(in) :: atom
         end subroutine
         subroutine atomic_ref_l8(value, atom)
         logical(8), intent(out) :: value
         logical, intent(in) :: atom
         end subroutine
      end interface

