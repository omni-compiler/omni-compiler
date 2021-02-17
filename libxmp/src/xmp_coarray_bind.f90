    subroutine xmp_coarray_bind_1d_r4(desc,a_decl)
    integer(8) desc
    real(4), pointer, intent(inout) :: a_decl(:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      real(4), pointer, intent(inout) :: a_decl(:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      real(4):: obj(l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      real(4), pointer :: p(:)
      real(4), target  :: d(:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_2d_r4(desc,a_decl)
    integer(8) desc
    real(4), pointer, intent(inout) :: a_decl(:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      real(4), pointer, intent(inout) :: a_decl(:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      real(4):: obj(l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      real(4), pointer :: p(:,:)
      real(4), target  :: d(:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_3d_r4(desc,a_decl)
    integer(8) desc
    real(4), pointer, intent(inout) :: a_decl(:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      real(4), pointer, intent(inout) :: a_decl(:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      real(4):: obj(l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      real(4), pointer :: p(:,:,:)
      real(4), target  :: d(:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_4d_r4(desc,a_decl)
    integer(8) desc
    real(4), pointer, intent(inout) :: a_decl(:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      real(4), pointer, intent(inout) :: a_decl(:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      real(4):: obj(l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      real(4), pointer :: p(:,:,:,:)
      real(4), target  :: d(:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_5d_r4(desc,a_decl)
    integer(8) desc
    real(4), pointer, intent(inout) :: a_decl(:,:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      real(4), pointer, intent(inout) :: a_decl(:,:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      real(4):: obj(l(5):u(5),l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      real(4), pointer :: p(:,:,:,:,:)
      real(4), target  :: d(:,:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_6d_r4(desc,a_decl)
    integer(8) desc
    real(4), pointer, intent(inout) :: a_decl(:,:,:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      real(4), pointer, intent(inout) :: a_decl(:,:,:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      real(4):: obj(l(6):u(6),l(5):u(5),l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      real(4), pointer :: p(:,:,:,:,:,:)
      real(4), target  :: d(:,:,:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_7d_r4(desc,a_decl)
    integer(8) desc
    real(4), pointer, intent(inout) :: a_decl(:,:,:,:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      real(4), pointer, intent(inout) :: a_decl(:,:,:,:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      real(4):: obj(l(7):u(7),l(6):u(6),l(5):u(5),l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      real(4), pointer :: p(:,:,:,:,:,:,:)
      real(4), target  :: d(:,:,:,:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_1d_r8(desc,a_decl)
    integer(8) desc
    real(8), pointer, intent(inout) :: a_decl(:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      real(8), pointer, intent(inout) :: a_decl(:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      real(8):: obj(l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      real(8), pointer :: p(:)
      real(8), target  :: d(:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_2d_r8(desc,a_decl)
    integer(8) desc
    real(8), pointer, intent(inout) :: a_decl(:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      real(8), pointer, intent(inout) :: a_decl(:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      real(8):: obj(l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      real(8), pointer :: p(:,:)
      real(8), target  :: d(:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_3d_r8(desc,a_decl)
    integer(8) desc
    real(8), pointer, intent(inout) :: a_decl(:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      real(8), pointer, intent(inout) :: a_decl(:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      real(8):: obj(l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      real(8), pointer :: p(:,:,:)
      real(8), target  :: d(:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_4d_r8(desc,a_decl)
    integer(8) desc
    real(8), pointer, intent(inout) :: a_decl(:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      real(8), pointer, intent(inout) :: a_decl(:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      real(8):: obj(l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      real(8), pointer :: p(:,:,:,:)
      real(8), target  :: d(:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_5d_r8(desc,a_decl)
    integer(8) desc
    real(8), pointer, intent(inout) :: a_decl(:,:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      real(8), pointer, intent(inout) :: a_decl(:,:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      real(8):: obj(l(5):u(5),l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      real(8), pointer :: p(:,:,:,:,:)
      real(8), target  :: d(:,:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_6d_r8(desc,a_decl)
    integer(8) desc
    real(8), pointer, intent(inout) :: a_decl(:,:,:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      real(8), pointer, intent(inout) :: a_decl(:,:,:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      real(8):: obj(l(6):u(6),l(5):u(5),l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      real(8), pointer :: p(:,:,:,:,:,:)
      real(8), target  :: d(:,:,:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_7d_r8(desc,a_decl)
    integer(8) desc
    real(8), pointer, intent(inout) :: a_decl(:,:,:,:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      real(8), pointer, intent(inout) :: a_decl(:,:,:,:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      real(8):: obj(l(7):u(7),l(6):u(6),l(5):u(5),l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      real(8), pointer :: p(:,:,:,:,:,:,:)
      real(8), target  :: d(:,:,:,:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_1d_z8(desc,a_decl)
    integer(8) desc
    complex(4), pointer, intent(inout) :: a_decl(:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      complex(4), pointer, intent(inout) :: a_decl(:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      complex(4):: obj(l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      complex(4), pointer :: p(:)
      complex(4), target  :: d(:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_2d_z8(desc,a_decl)
    integer(8) desc
    complex(4), pointer, intent(inout) :: a_decl(:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      complex(4), pointer, intent(inout) :: a_decl(:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      complex(4):: obj(l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      complex(4), pointer :: p(:,:)
      complex(4), target  :: d(:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_3d_z8(desc,a_decl)
    integer(8) desc
    complex(4), pointer, intent(inout) :: a_decl(:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      complex(4), pointer, intent(inout) :: a_decl(:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      complex(4):: obj(l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      complex(4), pointer :: p(:,:,:)
      complex(4), target  :: d(:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_4d_z8(desc,a_decl)
    integer(8) desc
    complex(4), pointer, intent(inout) :: a_decl(:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      complex(4), pointer, intent(inout) :: a_decl(:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      complex(4):: obj(l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      complex(4), pointer :: p(:,:,:,:)
      complex(4), target  :: d(:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_5d_z8(desc,a_decl)
    integer(8) desc
    complex(4), pointer, intent(inout) :: a_decl(:,:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      complex(4), pointer, intent(inout) :: a_decl(:,:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      complex(4):: obj(l(5):u(5),l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      complex(4), pointer :: p(:,:,:,:,:)
      complex(4), target  :: d(:,:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_6d_z8(desc,a_decl)
    integer(8) desc
    complex(4), pointer, intent(inout) :: a_decl(:,:,:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      complex(4), pointer, intent(inout) :: a_decl(:,:,:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      complex(4):: obj(l(6):u(6),l(5):u(5),l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      complex(4), pointer :: p(:,:,:,:,:,:)
      complex(4), target  :: d(:,:,:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_7d_z8(desc,a_decl)
    integer(8) desc
    complex(4), pointer, intent(inout) :: a_decl(:,:,:,:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      complex(4), pointer, intent(inout) :: a_decl(:,:,:,:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      complex(4):: obj(l(7):u(7),l(6):u(6),l(5):u(5),l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      complex(4), pointer :: p(:,:,:,:,:,:,:)
      complex(4), target  :: d(:,:,:,:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_1d_z16(desc,a_decl)
    integer(8) desc
    complex(8), pointer, intent(inout) :: a_decl(:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      complex(8), pointer, intent(inout) :: a_decl(:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      complex(8):: obj(l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      complex(8), pointer :: p(:)
      complex(8), target  :: d(:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_2d_z16(desc,a_decl)
    integer(8) desc
    complex(8), pointer, intent(inout) :: a_decl(:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      complex(8), pointer, intent(inout) :: a_decl(:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      complex(8):: obj(l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      complex(8), pointer :: p(:,:)
      complex(8), target  :: d(:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_3d_z16(desc,a_decl)
    integer(8) desc
    complex(8), pointer, intent(inout) :: a_decl(:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      complex(8), pointer, intent(inout) :: a_decl(:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      complex(8):: obj(l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      complex(8), pointer :: p(:,:,:)
      complex(8), target  :: d(:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_4d_z16(desc,a_decl)
    integer(8) desc
    complex(8), pointer, intent(inout) :: a_decl(:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      complex(8), pointer, intent(inout) :: a_decl(:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      complex(8):: obj(l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      complex(8), pointer :: p(:,:,:,:)
      complex(8), target  :: d(:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_5d_z16(desc,a_decl)
    integer(8) desc
    complex(8), pointer, intent(inout) :: a_decl(:,:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      complex(8), pointer, intent(inout) :: a_decl(:,:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      complex(8):: obj(l(5):u(5),l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      complex(8), pointer :: p(:,:,:,:,:)
      complex(8), target  :: d(:,:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_6d_z16(desc,a_decl)
    integer(8) desc
    complex(8), pointer, intent(inout) :: a_decl(:,:,:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      complex(8), pointer, intent(inout) :: a_decl(:,:,:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      complex(8):: obj(l(6):u(6),l(5):u(5),l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      complex(8), pointer :: p(:,:,:,:,:,:)
      complex(8), target  :: d(:,:,:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_7d_z16(desc,a_decl)
    integer(8) desc
    complex(8), pointer, intent(inout) :: a_decl(:,:,:,:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      complex(8), pointer, intent(inout) :: a_decl(:,:,:,:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      complex(8):: obj(l(7):u(7),l(6):u(6),l(5):u(5),l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      complex(8), pointer :: p(:,:,:,:,:,:,:)
      complex(8), target  :: d(:,:,:,:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_1d_i2(desc,a_decl)
    integer(8) desc
    integer(2), pointer, intent(inout) :: a_decl(:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      integer(2), pointer, intent(inout) :: a_decl(:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      integer(2):: obj(l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      integer(2), pointer :: p(:)
      integer(2), target  :: d(:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_2d_i2(desc,a_decl)
    integer(8) desc
    integer(2), pointer, intent(inout) :: a_decl(:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      integer(2), pointer, intent(inout) :: a_decl(:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      integer(2):: obj(l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      integer(2), pointer :: p(:,:)
      integer(2), target  :: d(:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_3d_i2(desc,a_decl)
    integer(8) desc
    integer(2), pointer, intent(inout) :: a_decl(:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      integer(2), pointer, intent(inout) :: a_decl(:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      integer(2):: obj(l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      integer(2), pointer :: p(:,:,:)
      integer(2), target  :: d(:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_4d_i2(desc,a_decl)
    integer(8) desc
    integer(2), pointer, intent(inout) :: a_decl(:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      integer(2), pointer, intent(inout) :: a_decl(:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      integer(2):: obj(l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      integer(2), pointer :: p(:,:,:,:)
      integer(2), target  :: d(:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_5d_i2(desc,a_decl)
    integer(8) desc
    integer(2), pointer, intent(inout) :: a_decl(:,:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      integer(2), pointer, intent(inout) :: a_decl(:,:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      integer(2):: obj(l(5):u(5),l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      integer(2), pointer :: p(:,:,:,:,:)
      integer(2), target  :: d(:,:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_6d_i2(desc,a_decl)
    integer(8) desc
    integer(2), pointer, intent(inout) :: a_decl(:,:,:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      integer(2), pointer, intent(inout) :: a_decl(:,:,:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      integer(2):: obj(l(6):u(6),l(5):u(5),l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      integer(2), pointer :: p(:,:,:,:,:,:)
      integer(2), target  :: d(:,:,:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_7d_i2(desc,a_decl)
    integer(8) desc
    integer(2), pointer, intent(inout) :: a_decl(:,:,:,:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      integer(2), pointer, intent(inout) :: a_decl(:,:,:,:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      integer(2):: obj(l(7):u(7),l(6):u(6),l(5):u(5),l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      integer(2), pointer :: p(:,:,:,:,:,:,:)
      integer(2), target  :: d(:,:,:,:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_1d_i4(desc,a_decl)
    integer(8) desc
    integer(4), pointer, intent(inout) :: a_decl(:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      integer(4), pointer, intent(inout) :: a_decl(:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      integer(4):: obj(l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      integer(4), pointer :: p(:)
      integer(4), target  :: d(:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_2d_i4(desc,a_decl)
    integer(8) desc
    integer(4), pointer, intent(inout) :: a_decl(:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      integer(4), pointer, intent(inout) :: a_decl(:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      integer(4):: obj(l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      integer(4), pointer :: p(:,:)
      integer(4), target  :: d(:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_3d_i4(desc,a_decl)
    integer(8) desc
    integer(4), pointer, intent(inout) :: a_decl(:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      integer(4), pointer, intent(inout) :: a_decl(:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      integer(4):: obj(l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      integer(4), pointer :: p(:,:,:)
      integer(4), target  :: d(:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_4d_i4(desc,a_decl)
    integer(8) desc
    integer(4), pointer, intent(inout) :: a_decl(:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      integer(4), pointer, intent(inout) :: a_decl(:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      integer(4):: obj(l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      integer(4), pointer :: p(:,:,:,:)
      integer(4), target  :: d(:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_5d_i4(desc,a_decl)
    integer(8) desc
    integer(4), pointer, intent(inout) :: a_decl(:,:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      integer(4), pointer, intent(inout) :: a_decl(:,:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      integer(4):: obj(l(5):u(5),l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      integer(4), pointer :: p(:,:,:,:,:)
      integer(4), target  :: d(:,:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_6d_i4(desc,a_decl)
    integer(8) desc
    integer(4), pointer, intent(inout) :: a_decl(:,:,:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      integer(4), pointer, intent(inout) :: a_decl(:,:,:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      integer(4):: obj(l(6):u(6),l(5):u(5),l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      integer(4), pointer :: p(:,:,:,:,:,:)
      integer(4), target  :: d(:,:,:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_7d_i4(desc,a_decl)
    integer(8) desc
    integer(4), pointer, intent(inout) :: a_decl(:,:,:,:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      integer(4), pointer, intent(inout) :: a_decl(:,:,:,:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      integer(4):: obj(l(7):u(7),l(6):u(6),l(5):u(5),l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      integer(4), pointer :: p(:,:,:,:,:,:,:)
      integer(4), target  :: d(:,:,:,:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_1d_i8(desc,a_decl)
    integer(8) desc
    integer(8), pointer, intent(inout) :: a_decl(:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      integer(8), pointer, intent(inout) :: a_decl(:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      integer(8):: obj(l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      integer(8), pointer :: p(:)
      integer(8), target  :: d(:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_2d_i8(desc,a_decl)
    integer(8) desc
    integer(8), pointer, intent(inout) :: a_decl(:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      integer(8), pointer, intent(inout) :: a_decl(:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      integer(8):: obj(l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      integer(8), pointer :: p(:,:)
      integer(8), target  :: d(:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_3d_i8(desc,a_decl)
    integer(8) desc
    integer(8), pointer, intent(inout) :: a_decl(:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      integer(8), pointer, intent(inout) :: a_decl(:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      integer(8):: obj(l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      integer(8), pointer :: p(:,:,:)
      integer(8), target  :: d(:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_4d_i8(desc,a_decl)
    integer(8) desc
    integer(8), pointer, intent(inout) :: a_decl(:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      integer(8), pointer, intent(inout) :: a_decl(:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      integer(8):: obj(l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      integer(8), pointer :: p(:,:,:,:)
      integer(8), target  :: d(:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_5d_i8(desc,a_decl)
    integer(8) desc
    integer(8), pointer, intent(inout) :: a_decl(:,:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      integer(8), pointer, intent(inout) :: a_decl(:,:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      integer(8):: obj(l(5):u(5),l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      integer(8), pointer :: p(:,:,:,:,:)
      integer(8), target  :: d(:,:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_6d_i8(desc,a_decl)
    integer(8) desc
    integer(8), pointer, intent(inout) :: a_decl(:,:,:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      integer(8), pointer, intent(inout) :: a_decl(:,:,:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      integer(8):: obj(l(6):u(6),l(5):u(5),l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      integer(8), pointer :: p(:,:,:,:,:,:)
      integer(8), target  :: d(:,:,:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
    subroutine xmp_coarray_bind_7d_i8(desc,a_decl)
    integer(8) desc
    integer(8), pointer, intent(inout) :: a_decl(:,:,:,:,:,:,:)
    integer(4), dimension(7) :: ub, lb
    integer(8) :: addr
    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)
    call bind_cray_pointer(a_decl,lb,ub,addr)
    return
  contains
    subroutine bind_cray_pointer(a_decl,l,u,addr)
      integer(8), pointer, intent(inout) :: a_decl(:,:,:,:,:,:,:)
      integer(4), dimension(7) :: l,u
      integer(8) :: addr
      integer(8):: obj(l(7):u(7),l(6):u(6),l(5):u(5),l(4):u(4),l(3):u(3),l(2):u(2),l(1):u(1))
      pointer (crayptr, obj)
      call xmp_assign_cray_pointer(crayptr,addr)
      call pointer_assign(a_decl, obj)
    end subroutine bind_cray_pointer
    subroutine pointer_assign(p, d)
      integer(8), pointer :: p(:,:,:,:,:,:,:)
      integer(8), target  :: d(:,:,:,:,:,:,:)
      p => d
      return 
    end subroutine pointer_assign
  end subroutine
