      interface zzz_images
         subroutine zzz_image_0(image)
           integer, intent(in) :: image
         end subroutine
         subroutine zzz_image_1(image, stat)
           integer, intent(in) :: image
           integer, intent(out) :: stat
         end subroutine
         subroutine zzz_image_2(image, stat, errmsg)
           integer, intent(in) :: image
           integer, intent(out) :: stat
           character(len=*), intent(out) :: errmsg
         end subroutine

         subroutine zzz_images_0(images)
           integer, intent(in) :: images(:)
         end subroutine
         subroutine zzz_images_1(images, stat)
           integer, intent(in) :: images(:)
           integer, intent(out) :: stat
         end subroutine
         subroutine zzz_images_2(images, stat, errmsg)
           integer, intent(in) :: images(:)
           integer, intent(out) :: stat
           character(len=*), intent(out) :: errmsg
         end subroutine

         subroutine zzz_images_all_0(aster)
           character(len=1), intent(in) :: aster
         end subroutine
         subroutine zzz_images_all_1(aster, stat)
           character(len=1), intent(in) :: aster
           integer, intent(out) :: stat
         end subroutine
         subroutine zzz_images_all_2(aster, stat, errmsg)
           character(len=1), intent(in) :: aster
           integer, intent(out) :: stat
           character(len=*), intent(out) :: errmsg
         end subroutine
      end interface

      call zzz_images(2.1)
    end
