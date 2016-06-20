!-----------------------------------------------------------------------
      interface xmpf_atomic_define_generic
!-----------------------------------------------------------------------
      subroutine xmpf_atomic_define_self_i4(atom, value)
        integer(4), intent(out) :: atom
        integer(4), intent(in)  :: value
      end subroutine

      subroutine xmpf_atomic_define_self_l4(atom, value)
        logical(4), intent(out) :: atom
        logical(4), intent(in)  :: value
      end subroutine

      subroutine xmpf_atomic_define_remote_i4(descptr, coindex, mold,   &
     &  src)
        integer(8), intent(in) :: descptr
        integer, intent(in) :: coindex
        integer(4), intent(in) :: mold, src
      end subroutine

      subroutine xmpf_atomic_define_remote_l4(descptr, coindex, mold,   &
     &  src)
        integer(8), intent(in) :: descptr
        integer, intent(in) :: coindex
        logical(4), intent(in) :: mold, src
      end subroutine

      end interface

!-----------------------------------------------------------------------
      interface xmpf_atomic_ref_generic
!-----------------------------------------------------------------------
      subroutine xmpf_atomic_ref_i4(value, atom)
        integer(4), intent(out) :: value
        integer(4), intent(in)  :: atom
      end subroutine

      subroutine xmpf_atomic_ref_l4(value, atom)
        logical(4), intent(out) :: value
        logical(4), intent(in)  :: atom
      end subroutine

      end interface
