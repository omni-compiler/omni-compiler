!! XMP/F Coarray Declarations 
!!
!! XMP/F translator automatically insert a USE statement:
!!      use 'xmpf_coarray_decl'
!! into the output code if any coarray features are used in
!! the input program.
!!

module xmpf_coarray_decl

!-------------------------------
!  coarray intrinsics
!-------------------------------

      interface
         integer function xmpf_image_index(descptr, coindexes)
           integer(8), intent(in) :: descptr
           integer, intent(in) :: coindexes(*)
         end function xmpf_image_index
      end interface

      interface xmpf_cobound
         !! restriction: kind must be 4.
         function xmpf_cobound_nodim(descptr, kind, lu, corank)         &
     &    result(bounds)
           integer(8), intent(in) :: descptr
           integer, intent(in) :: corank, lu, kind
           integer bounds(corank)           !! allocate here in Fortran
         end function xmpf_cobound_nodim
         !! restriction: kind must be 4.
         function xmpf_cobound_dim(descptr, dim, kind, lu, corank)      &
     &    result(bound)
           integer(8), intent(in) :: descptr
           integer, intent(in) :: corank, dim, lu, kind
           integer bound
         end function xmpf_cobound_dim
      end interface

!! transformation functions

      interface xmpf_this_image
         function xmpf_this_image_coarray_wrap(descptr, corank)         &
     &    result(image)
           integer(8), intent(in) :: descptr
           integer, intent(in) :: corank
           integer image(corank)
         end function xmpf_this_image_coarray_wrap
         function xmpf_this_image_coarray_dim(descptr, corank, dim)     &
     &    result(coindex)
           integer(8), intent(in) :: descptr
           integer, intent(in) :: corank, dim
           integer coindex
         end function xmpf_this_image_coarray_dim
      end interface


!-------------------------------
!  coarray runtime interface
!-------------------------------
!! synchronization
      include "xmp_coarray_sync.h"
!!!      include "xmp_coarray_sync_sxace.h"

!!! This is not useful definition because it cannot be found by 
!!! XMPenv.findVarIdent(...) and therefore will be double-defined.
!!      interface
!!        integer function xmpf_coarray_get_image_index                   &
!!     &    (descptr, corank, i1, i2, i3, i4, i5, i6, i7)
!!        integer(8), intent(in) :: descptr
!!        integer, intent(in) :: corank, i1
!!        integer, intent(in), optional :: i2, i3, i4, i5, i6, i7
!!        end function
!!      end interface

!! reference of coindexed objects
      include "xmp_coarray_get.h"

!! assignment statements to coindex variables
      include "xmp_coarray_put.h"

!! allocate statement
      include "xmp_coarray_alloc.h"

!! reduction subroutines
      include "xmp_coarray_reduction.h"

contains
  !! test
  subroutine xmpf_coarray_hello
    write(*,*) "Hello Coarray"
  end subroutine xmpf_coarray_hello

end module xmpf_coarray_decl


