    subroutine xmp_coarray_bind_r4(desc,v_decl)
    integer(8) desc
    real(4), pointer, intent(inout) :: v_decl
    integer(4), dimension(7) :: ub, lb ! dummy 
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(v_decl,addr)
    return
  contains
    subroutine bind_cray_pointer(v_decl,addr)
      real(4), pointer, intent(inout) :: v_decl
      integer(8) :: addr
      real(4):: obj
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(v_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      real(4), pointer :: p
      real(4), target  :: d
      p => d
      return 
    end subroutine pointer_assign
  end subroutine

    subroutine xmp_coarray_mem_put_r4(img_dims,remote_desc,v,status)
    integer(8), intent(in) :: remote_desc
    integer(4), intent(in) :: img_dims(*)
    integer(4), intent(out):: status
    real(4) :: v
    call xmp_coarray_mem_put_addr(img_dims,remote_desc,4,loc(v),status)
  end subroutine

    subroutine xmp_coarray_mem_get_r4(img_dims,remote_desc,v,status)
    integer(8), intent(in) :: remote_desc
    integer(4), intent(in) :: img_dims(*)
    integer(4), intent(out):: status
    real(4) :: v
    call xmp_coarray_mem_get_addr(img_dims,remote_desc,4,loc(v),status)
  end subroutine

    subroutine xmp_coarray_bind_r8(desc,v_decl)
    integer(8) desc
    real(8), pointer, intent(inout) :: v_decl
    integer(4), dimension(7) :: ub, lb ! dummy 
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(v_decl,addr)
    return
  contains
    subroutine bind_cray_pointer(v_decl,addr)
      real(8), pointer, intent(inout) :: v_decl
      integer(8) :: addr
      real(8):: obj
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(v_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      real(8), pointer :: p
      real(8), target  :: d
      p => d
      return 
    end subroutine pointer_assign
  end subroutine

    subroutine xmp_coarray_mem_put_r8(img_dims,remote_desc,v,status)
    integer(8), intent(in) :: remote_desc
    integer(4), intent(in) :: img_dims(*)
    integer(4), intent(out):: status
    real(8) :: v
    call xmp_coarray_mem_put_addr(img_dims,remote_desc,8,loc(v),status)
  end subroutine

    subroutine xmp_coarray_mem_get_r8(img_dims,remote_desc,v,status)
    integer(8), intent(in) :: remote_desc
    integer(4), intent(in) :: img_dims(*)
    integer(4), intent(out):: status
    real(8) :: v
    call xmp_coarray_mem_get_addr(img_dims,remote_desc,8,loc(v),status)
  end subroutine

    subroutine xmp_coarray_bind_z8(desc,v_decl)
    integer(8) desc
    complex(4), pointer, intent(inout) :: v_decl
    integer(4), dimension(7) :: ub, lb ! dummy 
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(v_decl,addr)
    return
  contains
    subroutine bind_cray_pointer(v_decl,addr)
      complex(4), pointer, intent(inout) :: v_decl
      integer(8) :: addr
      complex(4):: obj
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(v_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      complex(4), pointer :: p
      complex(4), target  :: d
      p => d
      return 
    end subroutine pointer_assign
  end subroutine

    subroutine xmp_coarray_mem_put_z8(img_dims,remote_desc,v,status)
    integer(8), intent(in) :: remote_desc
    integer(4), intent(in) :: img_dims(*)
    integer(4), intent(out):: status
    complex(4) :: v
    call xmp_coarray_mem_put_addr(img_dims,remote_desc,4,loc(v),status)
  end subroutine

    subroutine xmp_coarray_mem_get_z8(img_dims,remote_desc,v,status)
    integer(8), intent(in) :: remote_desc
    integer(4), intent(in) :: img_dims(*)
    integer(4), intent(out):: status
    complex(4) :: v
    call xmp_coarray_mem_get_addr(img_dims,remote_desc,4,loc(v),status)
  end subroutine

    subroutine xmp_coarray_bind_z16(desc,v_decl)
    integer(8) desc
    complex(8), pointer, intent(inout) :: v_decl
    integer(4), dimension(7) :: ub, lb ! dummy 
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(v_decl,addr)
    return
  contains
    subroutine bind_cray_pointer(v_decl,addr)
      complex(8), pointer, intent(inout) :: v_decl
      integer(8) :: addr
      complex(8):: obj
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(v_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      complex(8), pointer :: p
      complex(8), target  :: d
      p => d
      return 
    end subroutine pointer_assign
  end subroutine

    subroutine xmp_coarray_mem_put_z16(img_dims,remote_desc,v,status)
    integer(8), intent(in) :: remote_desc
    integer(4), intent(in) :: img_dims(*)
    integer(4), intent(out):: status
    complex(8) :: v
    call xmp_coarray_mem_put_addr(img_dims,remote_desc,8,loc(v),status)
  end subroutine

    subroutine xmp_coarray_mem_get_z16(img_dims,remote_desc,v,status)
    integer(8), intent(in) :: remote_desc
    integer(4), intent(in) :: img_dims(*)
    integer(4), intent(out):: status
    complex(8) :: v
    call xmp_coarray_mem_get_addr(img_dims,remote_desc,8,loc(v),status)
  end subroutine

    subroutine xmp_coarray_bind_i2(desc,v_decl)
    integer(8) desc
    integer(2), pointer, intent(inout) :: v_decl
    integer(4), dimension(7) :: ub, lb ! dummy 
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(v_decl,addr)
    return
  contains
    subroutine bind_cray_pointer(v_decl,addr)
      integer(2), pointer, intent(inout) :: v_decl
      integer(8) :: addr
      integer(2):: obj
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(v_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      integer(2), pointer :: p
      integer(2), target  :: d
      p => d
      return 
    end subroutine pointer_assign
  end subroutine

    subroutine xmp_coarray_mem_put_i2(img_dims,remote_desc,v,status)
    integer(8), intent(in) :: remote_desc
    integer(4), intent(in) :: img_dims(*)
    integer(4), intent(out):: status
    integer(2) :: v
    call xmp_coarray_mem_put_addr(img_dims,remote_desc,2,loc(v),status)
  end subroutine

    subroutine xmp_coarray_mem_get_i2(img_dims,remote_desc,v,status)
    integer(8), intent(in) :: remote_desc
    integer(4), intent(in) :: img_dims(*)
    integer(4), intent(out):: status
    integer(2) :: v
    call xmp_coarray_mem_get_addr(img_dims,remote_desc,2,loc(v),status)
  end subroutine

    subroutine xmp_coarray_bind_i4(desc,v_decl)
    integer(8) desc
    integer(4), pointer, intent(inout) :: v_decl
    integer(4), dimension(7) :: ub, lb ! dummy 
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(v_decl,addr)
    return
  contains
    subroutine bind_cray_pointer(v_decl,addr)
      integer(4), pointer, intent(inout) :: v_decl
      integer(8) :: addr
      integer(4):: obj
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(v_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      integer(4), pointer :: p
      integer(4), target  :: d
      p => d
      return 
    end subroutine pointer_assign
  end subroutine

    subroutine xmp_coarray_mem_put_i4(img_dims,remote_desc,v,status)
    integer(8), intent(in) :: remote_desc
    integer(4), intent(in) :: img_dims(*)
    integer(4), intent(out):: status
    integer(4) :: v
    call xmp_coarray_mem_put_addr(img_dims,remote_desc,4,loc(v),status)
  end subroutine

    subroutine xmp_coarray_mem_get_i4(img_dims,remote_desc,v,status)
    integer(8), intent(in) :: remote_desc
    integer(4), intent(in) :: img_dims(*)
    integer(4), intent(out):: status
    integer(4) :: v
    call xmp_coarray_mem_get_addr(img_dims,remote_desc,4,loc(v),status)
  end subroutine

    subroutine xmp_coarray_bind_i8(desc,v_decl)
    integer(8) desc
    integer(8), pointer, intent(inout) :: v_decl
    integer(4), dimension(7) :: ub, lb ! dummy 
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(v_decl,addr)
    return
  contains
    subroutine bind_cray_pointer(v_decl,addr)
      integer(8), pointer, intent(inout) :: v_decl
      integer(8) :: addr
      integer(8):: obj
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(v_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      integer(8), pointer :: p
      integer(8), target  :: d
      p => d
      return 
    end subroutine pointer_assign
  end subroutine

    subroutine xmp_coarray_mem_put_i8(img_dims,remote_desc,v,status)
    integer(8), intent(in) :: remote_desc
    integer(4), intent(in) :: img_dims(*)
    integer(4), intent(out):: status
    integer(8) :: v
    call xmp_coarray_mem_put_addr(img_dims,remote_desc,8,loc(v),status)
  end subroutine

    subroutine xmp_coarray_mem_get_i8(img_dims,remote_desc,v,status)
    integer(8), intent(in) :: remote_desc
    integer(4), intent(in) :: img_dims(*)
    integer(4), intent(out):: status
    integer(8) :: v
    call xmp_coarray_mem_get_addr(img_dims,remote_desc,8,loc(v),status)
  end subroutine

