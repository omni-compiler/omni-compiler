!-----------------------------------------------------------------------
      subroutine xmpf_sync_all_2(stat, errmsg)
!-----------------------------------------------------------------------
      integer, intent(out) :: stat
      character(len=*), intent(out) :: errmsg
      call xmpf_sync_all_1(stat)
      call xmpf_get_errmsg(errmsg, len(errmsg))
      end subroutine

!-----------------------------------------------------------------------
      subroutine xmpf_sync_memory_2(stat, errmsg)
!-----------------------------------------------------------------------
      integer, intent(out) :: stat
      character(len=*), intent(out) :: errmsg
      call xmpf_sync_memory_1(stat)
      call xmpf_get_errmsg(errmsg, len(errmsg))
      end subroutine

!-----------------------------------------------------------------------
      subroutine xmpf_sync_image_2(image, stat, errmsg)
!-----------------------------------------------------------------------
      integer, intent(in) :: image
      integer, intent(out) :: stat
      character(len=*), intent(out) :: errmsg
      call xmpf_sync_image_1(image, status)
      call xmpf_get_errmsg(errmsg, len(errmsg))
      end subroutine

!-----------------------------------------------------------------------
      subroutine xmpf_sync_images_0(images)
!-----------------------------------------------------------------------
      integer, intent(in) :: images(:)
      call xmpf_sync_images_0s(size(images), images, status)
      end subroutine

!-----------------------------------------------------------------------
      subroutine xmpf_sync_images_1(images, stat)
!-----------------------------------------------------------------------
      integer, intent(in) :: images(:)
      integer, intent(out) :: stat
      call xmpf_sync_images_1s(size(images), images, status)
      end subroutine

!-----------------------------------------------------------------------
      subroutine xmpf_sync_images_2(images, stat, errmsg)
!-----------------------------------------------------------------------
      integer, intent(in) :: images(:)
      integer, intent(out) :: stat
      character(len=*), intent(out) :: errmsg
      call xmpf_sync_images_1s(size(images), images, status)
      call xmpf_get_errmsg(errmsg, len(errmsg))
      end subroutine

!-----------------------------------------------------------------------
      subroutine xmpf_sync_images_all_2(aster, stat, errmsg)
!-----------------------------------------------------------------------
      character(len=1), intent(in) :: aster
      integer, intent(out) :: stat
      character(len=*), intent(out) :: errmsg
      integer msglen, status
      call xmpf_sync_images_all_1(status)
      call xmpf_get_errmsg(errmsg, len(errmsg))
      end subroutine


