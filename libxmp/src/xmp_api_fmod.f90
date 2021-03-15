module xmp_api
  ! type xmp_desc
  !    sequence
  !    integer*8 :: xmp_desc
  ! end type xmp_desc

  include "xmp_constant_fmod.f90"

  interface
     subroutine xmp_init_all
     end subroutine xmp_init_all

     subroutine xmp_finalize_all
     end subroutine xmp_finalize_all

     function xmp_this_image()
       integer xmp_this_image
     end function xmp_this_image

     function xmp_num_images()
       integer xmp_num_images
     end function xmp_num_images

     function xmp_all_node_num()
       integer xmp_all_node_num
     end function xmp_all_node_num

     function xmp_node_num()
       integer xmp_node_num
     end function xmp_node_num

     function xmp_wtime()
       real(8) xmp_wtime
     end function xmp_wtime

     function xmp_wtick()
       real(8) xmp_wtick
     end function xmp_wtick

     !!
     !! global view api
     !!
     subroutine xmp_global_nodes(desc, n_dims, dim_size, is_static)
       integer(8), intent(out):: desc
       integer(4), intent(in) :: n_dims, dim_size(*)
       logical, intent(in) ::  is_static
     end subroutine xmp_global_nodes

     subroutine xmp_new_template(desc, n_desc, n_dims, dim_lb, dim_ub)
       integer(8), intent(out) :: desc
       integer(8), intent(in) :: n_desc
       integer(4), intent(in) :: n_dims
       integer(8), dimension(*), intent(in) :: dim_lb, dim_ub
     end subroutine xmp_new_template

     subroutine xmp_dist_template_block(desc,template_dim_idx,node_dim_idx, status)
       integer(8), intent(in) :: desc
       integer(4), intent(in) :: template_dim_idx, node_dim_idx
       integer(4), intent(out) :: status
     end subroutine xmp_dist_template_block
     
     subroutine xmp_dist_template_cyclic(desc,template_dim_idx,node_dim_idx, status)
       integer(8), intent(in) :: desc
       integer(4), intent(in) :: template_dim_idx, node_dim_idx
       integer(4), intent(out) :: status
     end subroutine xmp_dist_template_cyclic

     subroutine xmp_dist_template_block_cyclic(desc,template_dim_idx,node_dim_idx, width,status)
       integer(8), intent(in) :: desc
       integer(4), intent(in) :: template_dim_idx, node_dim_idx, width
       integer(4), intent(out) :: status
     end subroutine xmp_dist_template_block_cyclic

     subroutine xmp_new_array(desc,t_desc, type, n_dims, dim_lb, dim_ub)
       integer(8), intent(out) :: desc
       integer(8), intent(in) :: t_desc
       integer(4), intent(in) :: type, n_dims
       integer(8), dimension(*), intent(in) :: dim_lb, dim_ub
     end subroutine xmp_new_array

     subroutine xmp_align_array(a_desc,array_dim_idx, template_dim_idx, offset, status)
       integer(8), intent(in) :: a_desc
       integer(4), intent(in) :: array_dim_idx, template_dim_idx
       integer(4), intent(in) :: offset
       integer(4), intent(out) :: status
     end subroutine xmp_align_array
     
     subroutine xmp_set_shadow(a_desc, dim_idx, shdw_size_lo, shdw_size_hi, status)
       integer(8), intent(in) :: a_desc
       integer(4), intent(in) :: dim_idx, shdw_size_lo, shdw_size_hi
       integer(4), intent(out) :: status
     end subroutine xmp_set_shadow

     subroutine xmp_set_full_shadow(a_desc, dim_idx, status)
       integer(8), intent(in) :: a_desc
       integer(4), intent(in) :: dim_idx
       integer(4), intent(out) :: status
     end subroutine xmp_set_full_shadow

     subroutine xmp_get_array_local_dim(a_desc, dim_lb, dim_ub, status)
       integer(8), intent(in) :: a_desc
       integer(4), dimension(*), intent(out) :: dim_lb, dim_ub
       integer(4), intent(out) :: status
     end subroutine xmp_get_array_local_dim
     
     subroutine xmp_allocate_array(a_desc, addr, status)
       integer(8), intent(in) :: a_desc
       integer(8), intent(in) :: addr
       integer(4), intent(out) :: status
     end subroutine xmp_allocate_array

     subroutine xmp_loop_schedule(ser_start, ser_end, ser_step, &
       t_desc, t_idx, par_start, par_end, par_step, status) 
       integer(8), intent(in) :: t_desc
       integer(4), intent(in) :: ser_start, ser_end, ser_step, t_idx
       integer(4), intent(out) :: par_start, par_end, par_step, status
     end subroutine xmp_loop_schedule

     subroutine xmp_template_ltog(t_desc, dim, local_idx, global_idx,status)
       integer(8), intent(in) :: t_desc
       integer(4), intent(in) :: dim
       integer(4), intent(in) :: local_idx
       integer(8), intent(out) :: global_idx
       integer(4), intent(out) :: status
     end subroutine xmp_template_ltog

     subroutine xmp_template_gtol(t_desc, dim, global_idx, local_idx, status)
       integer(8), intent(in) :: t_desc
       integer(4), intent(in) :: dim
       integer(8), intent(in) :: global_idx
       integer(4), intent(out) :: local_idx
       integer(4), intent(out) :: status
     end subroutine xmp_template_gtol

     subroutine xmp_array_reflect(a_desc, status)
       integer(8), intent(in) :: a_desc
       integer(4), intent(out) :: status
     end subroutine xmp_array_reflect

     subroutine xmp_reduction_scalar(kind, type, loc, status)
       integer(4), intent(in) :: kind, type
       integer(8), intent(in) :: loc
       integer(4), intent(out) :: status
     end subroutine xmp_reduction_scalar

     !!
     !! coarray api
     !!
     subroutine xmp_new_corray(desc,elmt_size, ndims, dim_lb, dim_ub, &
          img_ndims, img_dim_size)
!       type(xmp_desc), intent(out):: desc
       integer(8), intent(out):: desc
       integer(4), intent(in):: elmt_size,ndims,img_ndims, img_dim_size(*)
       integer(8), intent(in), dimension(*) :: dim_lb, dim_ub
     end subroutine xmp_new_corray

     subroutine xmp_new_corray_mem(desc, nbytes, img_ndims, img_dim_size)
!       type(xmp_desc), intent(out):: desc
       integer(8), intent(out):: desc
       integer(4), intent(in):: nbytes,img_ndims, img_dim_size(*)
     end subroutine xmp_new_corray_mem

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

     subroutine xmp_coarray_mem_put_addr(img_dims,remote_desc,nbytes,addr,status)
       integer(8), intent(in) :: remote_desc, addr
       integer(4), intent(in) :: img_dims(*), nbytes
       integer(4), intent(out):: status
     end subroutine xmp_coarray_mem_put_addr

     subroutine xmp_coarray_mem_get_addr(img_dims,remote_desc,nbytes,addr,status)
       integer(8), intent(in) :: remote_desc, addr
       integer(4), intent(in) :: img_dims(*), nbytes
       integer(4), intent(out):: status
     end subroutine xmp_coarray_mem_get_addr

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
       integer(8), intent(in), dimension(*):: dim_lb, dim_ub
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

!    module procedure xmp_coarray_bind_i2
     module procedure xmp_coarray_bind_i4
!     module procedure xmp_coarray_bind_i8
!     module procedure xmp_coarray_bind_r4
!     module procedure xmp_coarray_bind_r8
!     module procedure xmp_coarray_bind_z4
!     module procedure xmp_coarray_bind_z8
     module procedure xmp_coarray_bind_char
          
  end interface xmp_coarray_bind

  interface xmp_coarray_mem_put
     module procedure xmp_coarray_mem_put_char
!     module procedure xmp_coarray_mem_put_i2
     module procedure xmp_coarray_mem_put_i4
!     module procedure xmp_coarray_mem_put_i8
!     module procedure xmp_coarray_mem_put_r4
!     module procedure xmp_coarray_mem_put_r8
!     module procedure xmp_coarray_mem_put_z4
!     module procedure xmp_coarray_mem_put_z8
  end interface xmp_coarray_mem_put

  interface xmp_coarray_mem_get
     module procedure xmp_coarray_mem_get_char
!     module procedure xmp_coarray_mem_get_i2
     module procedure xmp_coarray_mem_get_i4
!     module procedure xmp_coarray_mem_get_i8
!     module procedure xmp_coarray_mem_get_r4
!     module procedure xmp_coarray_mem_get_r8
!     module procedure xmp_coarray_mem_get_z4
!     module procedure xmp_coarray_mem_get_z8
  end interface xmp_coarray_mem_get

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

  include "xmp_coarray_bind_scalar.f90"
  ! subroutine xmp_coarray_bind_i4(desc,v_decl)
  !   integer(8) desc
  !   integer(4), pointer, intent(inout) :: v_decl
  !   integer(8) :: addr  ! address
  !   integer(4), dimension(7) :: ub, lb ! not used
  !   call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
  !   call bind_cray_pointer(v_decl,addr)
  !   return
  ! contains
  !   subroutine bind_cray_pointer(v_decl,addr)
  !     integer(4), pointer, intent(inout) :: v_decl
  !     integer(4) :: obj
  !     integer(8) :: addr  ! address
  !     pointer (crayptr, obj)
  !     call xmp_assign_cray_pointer(crayptr,addr)
  !     call pointer_assign(v_decl, obj)
  !   end subroutine bind_cray_pointer
  !   subroutine pointer_assign(p, d)
  !     integer(4), pointer :: p
  !     integer(4), target  :: d
  !     p => d
  !     return 
  !   end subroutine pointer_assign
  ! end subroutine 

  ! subroutine xmp_coarray_mem_put_i4(img_dims,remote_desc,v,status)
  !   integer(8), intent(in) :: remote_desc
  !   integer(4), intent(in) :: img_dims(*)
  !   integer(4), intent(out):: status
  !   integer(4) :: v
  !   call xmp_coarray_mem_put_addr(img_dims,remote_desc,4,loc(v),status)
  ! end subroutine xmp_coarray_mem_put_i4

  ! subroutine xmp_coarray_mem_get_i4(img_dims,remote_desc,v,status)
  !   integer(8), intent(in) :: remote_desc
  !   integer(4), intent(in) :: img_dims(*)
  !   integer(4), intent(out):: status
  !   integer(4) :: v
  !   call xmp_coarray_mem_get_addr(img_dims,remote_desc,4,loc(v),status)
  ! end subroutine xmp_coarray_mem_get_i4

  subroutine xmp_coarray_bind_char(desc,v_decl)
    integer(8) desc
    character(len=*), pointer, intent(inout) :: v_decl
    integer(8) :: addr  ! address
    integer(4), dimension(7) :: ub, lb ! not used
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(v_decl,addr)
    return
  contains
    subroutine bind_cray_pointer(v_decl,addr)
      character(len=*), pointer, intent(inout) :: v_decl
      character(len=1) :: obj
      integer(8) :: addr  ! address
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(v_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      character(len=*), pointer :: p
      character(len=*), target  :: d
      p => d
      return 
    end subroutine pointer_assign
  end subroutine 

  subroutine xmp_coarray_mem_put_char(img_dims,remote_desc, s, nbytes, status)
    integer(8), intent(in) :: remote_desc
    integer(4), intent(in) :: img_dims(*), nbytes
    integer(4), intent(out):: status
    character(len=*) :: s
    call xmp_coarray_mem_put_addr(img_dims,remote_desc,nbytes,loc(s),status)
  end subroutine xmp_coarray_mem_put_char

  subroutine xmp_coarray_mem_get_char(img_dims,remote_desc, s, nbytes, status)
    integer(8), intent(in) :: remote_desc
    integer(4), intent(in) :: img_dims(*), nbytes
    integer(4), intent(out):: status
    character(len=*) :: s
    call xmp_coarray_mem_get_addr(img_dims,remote_desc,nbytes,loc(s),status)
  end subroutine xmp_coarray_mem_get_char

end module xmp_api

