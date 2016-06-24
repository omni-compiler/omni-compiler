!-----------------------------------------------------------------------
!   atomic_define
!     declared in ../include/xmp_coarray_atomic.h:xmpf_atomic_define_generic
!
!   Argument mold, which is the local array element corresponding to the
!   target array element, is intentionally exposed to the native Fortran
!   compiler to suppress excessive code motion optimization.
!
!-----------------------------------------------------------------------

!!      subroutine xmpf_atomic_define_self_i4(descptr, atom, src)
!!      subroutine xmpf_atomic_define_self_l4(descptr, atom, src)


!!      subroutine xmpf_atomic_define_remote_i4(descptr, coindex, mold,   &
!!     &  src)
!!        integer(8), intent(in) :: descptr
!!        integer, intent(in) :: coindex
!!        integer, intent(inout) :: mold
!!        integer, intent(in) :: src
!!
!!        call xmpf_coarray_put_scalar(descptr, loc(mold), 4,             &
!!     &    coindex, src, 1)       !! 1: synchronous (wait for ack.)
!!        return
!!      end subroutine
!!
!!      subroutine xmpf_atomic_define_remote_l4(descptr, coindex, mold,   &
!!     &  src)
!!        integer(8), intent(in) :: descptr
!!        integer, intent(in) :: coindex
!!        logical, intent(inout) :: mold
!!        logical, intent(in) :: src
!!
!!        call xmpf_coarray_put_scalar(descptr, loc(mold), 4,             &
!!     &    coindex, src, 1)       !! 1: synchronous (wait for ack.)
!!        return
!!      end subroutine


!-----------------------------------------------------------------------
!   atomic_ref
!     declared in ../include/xmp_coarray_atomic.h:xmpf_atomic_ref_generic
!-----------------------------------------------------------------------

!! subroutine xmpf_atomic_ref_self_i4(descptr, atom, dst)
!! subroutine xmpf_atomic_ref_self_l4(descptr, atom, dst)

!!  subroutine xmpf_atomic_ref_remote_i4(descptr, coindex, mold, dst)
!!  subroutine xmpf_atomic_ref_remote_l4(descptr, coindex, mold, dst)
