!-----------------------------------------------------------------------
!   atomic_define
!     declared in ../include/xmp_coarray_atomic.h:xmpf_atomic_define_generic
!-----------------------------------------------------------------------
      subroutine xmpf_atomic_define_self_i4(atom, value)
        integer(4), intent(out) :: atom
        integer(4), intent(in)  :: value
        atom = value
      end subroutine

      subroutine xmpf_atomic_define_self_l4(atom, value)
        logical(4), intent(out) :: atom
        logical(4), intent(in)  :: value
        atom = value
      end subroutine

      subroutine xmpf_atomic_define_remote_i4(descptr, coindex, mold,   &
     &  src)
        integer(8), intent(in) :: descptr
        integer, intent(in) :: coindex
        integer(4), intent(in) :: mold, src

        call xmpf_coarray_put_scalar(descptr, loc(mold), 4,             &
     &    coindex, src, 1)       !! 1: synchronous (wait for ack.)
        return
      end subroutine

      subroutine xmpf_atomic_define_remote_l4(descptr, coindex, mold,   &
     &  src)
        integer(8), intent(in) :: descptr
        integer, intent(in) :: coindex
        logical(4), intent(in) :: mold, src

        call xmpf_coarray_put_scalar(descptr, loc(mold), 4,             &
     &    coindex, src, 1)       !! 1: synchronous (wait for ack.)
        return
      end subroutine


!-----------------------------------------------------------------------
!   atomic_ref
!     declared in ../include/xmp_coarray_atomic.h:xmpf_atomic_ref_generic
!-----------------------------------------------------------------------
      subroutine xmpf_atomic_ref_i4(value, atom)
        integer(4), intent(out) :: value
        integer(4), intent(in)  :: atom
        value = atom
      end subroutine

      subroutine xmpf_atomic_ref_l4(value, atom)
        logical(4), intent(out) :: value
        logical(4), intent(in)  :: atom
        value = atom
      end subroutine

!! no need
!!      subroutine xmpf_atomic_ref_remote_i4(descptr, coindex, mold,      &
!!     &  dst)
!!        integer(8), intent(in) :: descptr
!!        integer, intent(in) :: coindex
!!        integer(4), intent(in) :: mold, dst
!!        call xmpf_coarray_get_scalar(descptr, loc(mold), 4,             &
!!     &    coindex, dst)
!!        return
!!      end subroutine
!!
!!      subroutine xmpf_atomic_ref_remote_l4(descptr, coindex, mold,      &
!!     &  dst)
!!        integer(8), intent(in) :: descptr
!!        integer, intent(in) :: coindex
!!        logical(4), intent(in) :: mold, dst
!!        call xmpf_coarray_get_scalar(descptr, loc(mold), 4,             &
!!     &    coindex, dst)
!!        return
!!      end subroutine
