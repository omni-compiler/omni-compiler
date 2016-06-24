!-----------------------------------------------------------------------
      interface xmpf_atomic_define_generic
!-----------------------------------------------------------------------
      subroutine xmpf_atomic_define_self_i4(descptr, atom, src)
        integer(8), intent(in) :: descptr
        integer, intent(out) :: atom
        integer, intent(in) :: src
      end subroutine

      subroutine xmpf_atomic_define_self_l4(descptr, atom, src)
        integer(8), intent(in) :: descptr
        logical, intent(out) :: atom
        logical, intent(in) :: src
      end subroutine

      subroutine xmpf_atomic_define_remote_i4(descptr,coindex,mold,src)
        integer(8), intent(in) :: descptr
        integer, intent(in) :: coindex
        integer, intent(inout) :: mold   !! fake intent to suppress excess code motion
        integer, intent(in) :: src
      end subroutine

      subroutine xmpf_atomic_define_remote_l4(descptr,coindex,mold,src)
        integer(8), intent(in) :: descptr
        integer, intent(in) :: coindex
        logical, intent(inout) :: mold   !! fake intent to suppress excess code motion
        logical, intent(in) :: src
      end subroutine

      end interface

!-----------------------------------------------------------------------
      interface xmpf_atomic_ref_generic
!-----------------------------------------------------------------------
      subroutine xmpf_atomic_ref_self_i4(descptr, atom, dst)
        integer(8), intent(in) :: descptr
        integer, intent(in) :: atom
        integer, intent(out) :: dst
      end subroutine

      subroutine xmpf_atomic_ref_self_l4(descptr, atom, dst)
        integer(8), intent(in) :: descptr
        logical, intent(in) :: atom
        logical, intent(out) :: dst
      end subroutine

      subroutine xmpf_atomic_ref_remote_i4(descptr, coindex, mold, dst)
        integer(8), intent(in) :: descptr
        integer, intent(in) :: coindex
        integer, intent(inout) :: mold   !! fake intent to suppress excess code motion
        integer, intent(out) :: dst
      end subroutine

      subroutine xmpf_atomic_ref_remote_l4(descptr, coindex, mold, dst)
        integer(8), intent(in) :: descptr
        integer, intent(in) :: coindex
        logical, intent(inout) :: mold   !! fake intent to suppress excess code motion
        logical, intent(out) :: dst
      end subroutine

      end interface
