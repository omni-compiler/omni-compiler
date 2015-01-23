!-----------------------------------------------------------------------
      integer function xmp_num_images()
!-----------------------------------------------------------------------
      return xmpf_num_images()
      end function

!-----------------------------------------------------------------------
      integer function xmp_this_image()
!-----------------------------------------------------------------------
      return xmpf_this_image()
      end function

!-----------------------------------------------------------------------
      subroutine xmp_sync_all(stat, errmsg)
      integer, optional, intent(out) :: stat
      character(len=*), optional, intent(out) :: errmsg
!-----------------------------------------------------------------------
      integer msglen, status

      call xmpf_sync_all(status)

      if (present(stat)) then
         stat = status
      end if
      
      if (present(errmsg)) then
         call xmpf_get_errmsg(errmsg, len(errmsg))
      end if

      end subroutine

!-----------------------------------------------------------------------
      subroutine xmp_sync_memory(stat, errmsg)
      integer, optional, intent(out) :: stat
      character(len=*), optional, intent(out) :: errmsg
!-----------------------------------------------------------------------
      integer msglen, status

      call xmpf_sync_memory(status)

      if (present(stat)) then
         stat = status
      end if
      
      if (present(errmsg)) then
         call xmpf_get_errmsg(errmsg, len(errmsg))
      end if

      end subroutine

!-----------------------------------------------------------------------
      subroutine xmp_sync_images_one(image, stat, errmsg)
      integer, intent(in) :: image
      integer, optional, intent(out) :: stat
      character(len=*), optional, intent(out) :: errmsg
!-----------------------------------------------------------------------
      integer msglen, status

      call xmpf_sync_images_one(image, status)

      if (present(stat)) then
         stat = status
      end if
      
      if (present(errmsg)) then
         call xmpf_get_errmsg(errmsg, len(errmsg))
      end if

      end subroutine

!-----------------------------------------------------------------------
      subroutine xmp_sync_images_ast(image, stat, errmsg)
      character(len=1), intent(in) :: image
      integer, optional, intent(out) :: stat
      character(len=*), optional, intent(out) :: errmsg
!-----------------------------------------------------------------------
      integer msglen, status

      call xmpf_sync_images_ast(status)

      if (present(stat)) then
         stat = status
      end if
      
      if (present(errmsg)) then
         call xmpf_get_errmsg(errmsg, len(errmsg))
      end if

      end subroutine


