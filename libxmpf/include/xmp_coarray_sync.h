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
      interface
  !!         subroutine xmpf_sync_memory(arg1, ...)
  !!           class(*) :: arg1
  !!         end subroutine xmpf_sync_memory

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
         end subroutine xmpf_sync_image_nostat
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
!!      interface
!!         subroutine xmpf_lock(lock_var, stat, errmsg)
!!         type(lock_type) :: lock_var
!!         integer, optional, intent(out) :: stat
!!         character(len=*), optional, intent(out) :: errmsg
!!         end subroutine
!!      end interface
!!
!!      interface
!!         subroutine xmpf_unlock(lock_var, stat, errmsg)
!!         type(lock_type) :: lock_var
!!         integer, optional, intent(out) :: stat
!!         character(len=*), optional, intent(out) :: errmsg
!!         end subroutine
!!      end interface

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

