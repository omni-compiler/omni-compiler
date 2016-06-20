!-----------------------------------------------------------------------
!   array function this_image(coarray)
!     declared in ./xmpf_coarray_decl.f90:xmpf_this_image_generic
!-----------------------------------------------------------------------
      function xmpf_this_image_coarray_wrap(descptr, corank) result(image)
        integer(8), intent(in) :: descptr
        integer, intent(in) :: corank
        integer image(corank)           !! allocate here in Fortran

        call xmpf_this_image_coarray(descptr, corank, image)
        return
      end function xmpf_this_image_coarray_wrap


!-----------------------------------------------------------------------
!   array function lcobound/ucobound(coarray, kind)
!     declared in ./xmpf_coarray_decl.f90:xmpf_cobound_generic
!-----------------------------------------------------------------------
      function xmpf_cobound_nodim(descptr, kind, lu, corank)            &
     & result(bounds)
        integer(8), intent(in) :: descptr
        integer, intent(in) :: corank, lu, kind
        integer bounds(corank)           !! allocate here in Fortran

        call xmpf_cobound_nodim_subr(descptr, kind, lu, corank, bounds)
        return
      end function xmpf_cobound_nodim


!-----------------------------------------------------------------------
!   sync all
!     declared in ../include/xmp_coarray_sync.h
!-----------------------------------------------------------------------
      subroutine xmpf_sync_all_stat(stat, errmsg)
        integer, intent(out) :: stat
        character(len=*), intent(out), optional :: errmsg
        character(len=4) :: dummy

        if (present(errmsg)) then
           call xmpf_sync_all_stat_core(stat, errmsg, len(errmsg))
        else
           call xmpf_sync_all_stat_core(stat, dummy, 0)
        endif
      end subroutine xmpf_sync_all_stat

!-----------------------------------------------------------------------
!   sync memory
!     declared in ../include/xmp_coarray_sync.h:xmpf_sync_memory
!-----------------------------------------------------------------------
      subroutine xmpf_sync_memory_stat_wrap(stat, errmsg)
        integer, intent(out) :: stat
        character(len=*), intent(out), optional :: errmsg
        character(len=4) :: dummy

        if (present(errmsg)) then
           call xmpf_sync_memory_stat(stat, errmsg, len(errmsg))
        else
           call xmpf_sync_memory_stat(stat, dummy, 0)
        endif
      end subroutine

!-----------------------------------------------------------------------
!   sync images
!     declared in ../include/xmp_coarray_sync.h:xmpf_sync_images
!-----------------------------------------------------------------------

!!    no xmpf_sync_image_nostat_wrap(image)

      subroutine xmpf_sync_images_nostat_wrap(images)
        integer, intent(in) :: images(:)
        call xmpf_sync_images_nostat(images, size(images))
      end subroutine xmpf_sync_images_nostat_wrap

      subroutine xmpf_sync_allimages_nostat_wrap(aster)
        character(len=1), intent(in) :: aster
        call xmpf_sync_allimages_nostat()
      end subroutine xmpf_sync_allimages_nostat_wrap

      subroutine xmpf_sync_image_stat_wrap(image, stat, errmsg)
        integer, intent(in) :: image
        integer, intent(out) :: stat
        character(len=*), intent(out), optional :: errmsg
        character(len=4) :: dummy

        if (present(errmsg)) then
           call xmpf_sync_image_stat(image, stat, errmsg,               &
     &          len(errmsg))
        else
           call xmpf_sync_image_stat(image, stat, dummy, 0)
        endif
      end subroutine xmpf_sync_image_stat_wrap

      subroutine xmpf_sync_images_stat_wrap(images, stat, errmsg)
        integer, intent(in) :: images(:)
        integer, intent(out) :: stat
        character(len=*), intent(out), optional :: errmsg
        character(len=4) :: dummy

        if (present(errmsg)) then
           call xmpf_sync_images_stat(images, size(images), stat,       &
     &          errmsg, len(errmsg))
        else
           call xmpf_sync_images_stat(images, size(images), stat,       &
     &          dummy, 0)
        endif
      end subroutine xmpf_sync_images_stat_wrap

      subroutine xmpf_sync_allimages_stat_wrap(aster, stat, errmsg)
        character(len=1), intent(in) :: aster
        integer, intent(out) :: stat
        character(len=*), intent(out), optional :: errmsg
        character(len=4) :: dummy

        if (present(errmsg)) then
           call xmpf_sync_allimages_stat(stat, errmsg, len(errmsg))
        else
           call xmpf_sync_allimages_stat(stat, dummy, 0)
        endif
      end subroutine xmpf_sync_allimages_stat_wrap


!-----------------------------------------------------------------------
!   fatal error
!-----------------------------------------------------------------------

      subroutine xmpf_coarray_fatal(msg)
        character(*) :: msg
        call xmpf_coarray_fatal_with_len(msg, len(msg))
      end subroutine xmpf_coarray_fatal

