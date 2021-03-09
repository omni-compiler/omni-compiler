module xmp_api
  ! type xmp_desc
  !    sequence
  !    integer*8 :: xmp_desc
  ! end type xmp_desc

  interface
     subroutine xmp_init_all
     end subroutine xmp_init_all

     subroutine xmp_finalize_all
     end subroutine xmp_finalize_all

     function xmp_this_image()
       integer xmp_this_image
     end function xmp_this_image

     subroutine xmp_new_corray(desc,elmt_size, ndims, dim_lb, dim_ub, &
          img_ndims, img_dim_size)
!       type(xmp_desc), intent(out):: desc
       integer(8), intent(out):: desc
       integer(4), intent(in):: elmt_size,ndims,img_ndims, img_dim_size(*)
       integer(8), intent(in), dimension(*) :: dim_lb, dim_ub
     end subroutine xmp_new_corray

     subroutine xmp_coarray_deallocate(desc, status)
!       type(xmp_desc), intent(in):: desc
       integer(8), intent(in):: desc
       integer(4), intent(out):: status
     end subroutine xmp_coarray_deallocate

     subroutine xmp_coarray_get(img_dims,remote_desc,remote_asec, &
          local_desc, local_asec, status)
!       type(xmp_desc), intent(in) :: remote_desc, remote_asec, local_desc, local_asec
       integer(8), intent(in) :: remote_desc, remote_asec, local_desc, local_asec
       integer(4), intent(in) :: img_dims(*)
       integer(4), intent(out):: status
     end subroutine xmp_coarray_get

     subroutine xmp_coarray_get_local(img_dims,remote_desc,remote_asec, &
          local_desc, local_asec, status)
!       type(xmp_desc), intent(in) :: remote_desc, remote_asec, local_desc, local_asec
       integer(8), intent(in) :: remote_desc, remote_asec, local_desc, local_asec
       integer(4), intent(in) :: img_dims(*)
       integer(4), intent(out):: status
     end subroutine xmp_coarray_get_local

     subroutine xmp_coarray_put(img_dims,remote_desc,remote_asec, &
          local_desc, local_asec, status)
!       type(xmp_desc), intent(in) :: remote_desc, remote_asec, local_desc, local_asec
       integer(8), intent(in) :: remote_desc, remote_asec, local_desc, local_asec
       integer(4), intent(in) :: img_dims(*)
       integer(4), intent(out):: status
     end subroutine xmp_coarray_put

     subroutine xmp_coarray_put_local(img_dims,remote_desc,remote_asec, &
          local_desc, local_asec, status)
!       type(xmp_desc), intent(in) :: remote_desc, remote_asec, local_desc, local_asec
       integer(8), intent(in) :: remote_desc, remote_asec, local_desc, local_asec
       integer(4), intent(in) :: img_dims(*)
       integer(4), intent(out):: status
     end subroutine xmp_coarray_put_local

     subroutine xmp_new_array_section(desc, ndims)
!       type(xmp_desc), intent(out):: desc
       integer(8), intent(out):: desc
       integer(4), intent(in):: ndims
     end subroutine xmp_new_array_section

     subroutine xmp_free_array_section(desc)
!       type(xmp_desc), intent(in):: desc
       integer(8), intent(in):: desc
     end subroutine xmp_free_array_section

     subroutine xmp_array_section_set_info(desc,idx,start,end, status)
!       type(xmp_desc), intent(in):: desc
       integer(8), intent(in):: desc
       integer(4), intent(in):: idx
       integer(8), intent(in):: start, end
       integer(4), intent(out):: status
     end subroutine xmp_array_section_set_info

     subroutine xmp_array_section_set_triplet(desc,idx,start,end,stride,status)
!       type(xmp_desc), intent(in):: desc
       integer(8), intent(in):: desc
       integer(4), intent(in):: idx,stride
       integer(8), intent(in):: start, end
       integer(4), intent(out):: status
     end subroutine xmp_array_section_set_triplet

     subroutine xmp_new_local_array(desc, elem_size, n_dims, dim_lb, dim_ub, addr)
!       type(xmp_desc), intent(in):: desc
       integer(8), intent(out):: desc
       integer(4), intent(in):: elem_size, n_dims
       integer(8), intent(in):: dim_lb(:), dim_ub(:)
       integer(8), intent(in):: addr
     end subroutine xmp_new_local_array
       
     subroutine xmp_free_local_array(desc)
!       type(xmp_desc), intent(in):: desc
       integer(8), intent(in):: desc
     end subroutine xmp_free_local_array

     subroutine xmp_sync_all(status)
       integer(4) status
     end subroutine xmp_sync_all

     subroutine xmp_sync_memory(status)
       integer(4) status
     end subroutine xmp_sync_memory

     ! void xmp_sync_image_(int *_image, int *status)
     subroutine xmp_sync_image(image,status)
       integer(4) :: image, status
     end subroutine xmp_sync_image

     !void xmp_sync_images_(int *_num, int *image_set, int *status)
     subroutine xmp_sync_images(num, image_set, status)
       integer(4) :: num, image_set(:), status
     end subroutine xmp_sync_images

     !void xmp_sync_images_all(int *status)
     subroutine xmp_sync_images_all(status)
       integer(4) status
     end subroutine xmp_sync_images_all

  end interface

  interface xmp_coarray_bind
     module procedure xmp_coarray_bind_1d_r8
     module procedure xmp_coarray_bind_2d_r8
     module procedure xmp_coarray_bind_3d_r8
     module procedure xmp_coarray_bind_4d_r8
     module procedure xmp_coarray_bind_5d_r8
     module procedure xmp_coarray_bind_6d_r8
     module procedure xmp_coarray_bind_7d_r8

     module procedure xmp_coarray_bind_1d_r4
     module procedure xmp_coarray_bind_2d_r4
     module procedure xmp_coarray_bind_3d_r4
     module procedure xmp_coarray_bind_4d_r4
     module procedure xmp_coarray_bind_5d_r4
     module procedure xmp_coarray_bind_6d_r4
     module procedure xmp_coarray_bind_7d_r4

     module procedure xmp_coarray_bind_1d_z8
     module procedure xmp_coarray_bind_2d_z8
     module procedure xmp_coarray_bind_3d_z8
     module procedure xmp_coarray_bind_4d_z8
     module procedure xmp_coarray_bind_5d_z8
     module procedure xmp_coarray_bind_6d_z8
     module procedure xmp_coarray_bind_7d_z8

     module procedure xmp_coarray_bind_1d_z16
     module procedure xmp_coarray_bind_2d_z16
     module procedure xmp_coarray_bind_3d_z16
     module procedure xmp_coarray_bind_4d_z16
     module procedure xmp_coarray_bind_5d_z16
     module procedure xmp_coarray_bind_6d_z16
     module procedure xmp_coarray_bind_7d_z16

     module procedure xmp_coarray_bind_1d_i2
     module procedure xmp_coarray_bind_2d_i2
     module procedure xmp_coarray_bind_3d_i2
     module procedure xmp_coarray_bind_4d_i2
     module procedure xmp_coarray_bind_5d_i2
     module procedure xmp_coarray_bind_6d_i2
     module procedure xmp_coarray_bind_7d_i2

     module procedure xmp_coarray_bind_1d_i4
     module procedure xmp_coarray_bind_2d_i4
     module procedure xmp_coarray_bind_3d_i4
     module procedure xmp_coarray_bind_4d_i4
     module procedure xmp_coarray_bind_5d_i4
     module procedure xmp_coarray_bind_6d_i4
     module procedure xmp_coarray_bind_7d_i4

     module procedure xmp_coarray_bind_1d_i8
     module procedure xmp_coarray_bind_2d_i8
     module procedure xmp_coarray_bind_3d_i8
     module procedure xmp_coarray_bind_4d_i8
     module procedure xmp_coarray_bind_5d_i8
     module procedure xmp_coarray_bind_6d_i8
     module procedure xmp_coarray_bind_7d_i8

  end interface xmp_coarray_bind

contains
  include "xmp_coarray_bind.f90"
  ! subroutine xmp_coarray_bind_2d_i4(desc,a_decl)
  !   integer(8) desc
  !   integer(4), pointer, intent(inout) :: a_decl (:,:)
  !   integer(4), dimension(7) :: ub, lb
  !   integer(8) :: addr  ! address
  !   call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
  !   call bind_cray_pointer(a_decl,lb(2),ub(2),lb(1),ub(1),addr)
  !   return
  ! contains
  !   subroutine bind_cray_pointer(a_decl,lb1,ub1,lb2,ub2,addr)
  !     integer(4), pointer, intent(inout) :: a_decl (:,:)
  !     integer(4) :: ub1, lb1, ub2, lb2
  !     integer(8) :: addr  ! address
  !     integer(4) ::  obj(lb1:ub1,lb2:ub2)
  !     pointer (crayptr, obj)
  !     call xmp_assign_cray_pointer(crayptr,addr)
  !     call pointer_assign(a_decl, obj)
  !   end subroutine bind_cray_pointer
  !   subroutine pointer_assign(p, d)
  !     integer(4), pointer :: p(:,:)
  !     integer(4), target  :: d(:,:)
  !     p => d
  !     return 
  !   end subroutine pointer_assign
  ! end subroutine 
end module xmp_api

