!! XMP/F Coarray Declarations 
!!
!! This file is use-associated to the user program via the USE
!! statement inserted automatically.
!!
!! Besides this file is referred from a shellscript in backend:
!!   F-BackEnd/src/xcodeml/f/decompile/
!!     XfDecompileDomVisitor_coarrayLibs.java.sh
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

      interface xmpf_cobound_generic
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

      interface xmpf_num_images_generic
         integer function xmpf_num_images_current()
         end function
      end interface

      interface xmpf_this_image_generic
         function xmpf_this_image_current() result(image)
           integer image
         end function
         function xmpf_this_image_coarray_wrap(descptr, corank)         &
     &    result(image)
           integer(8), intent(in) :: descptr
           integer, intent(in) :: corank
           integer image(corank)
         end function
         function xmpf_this_image_coarray_dim(descptr, corank, dim)     &
     &    result(coindex)
           integer(8), intent(in) :: descptr
           integer, intent(in) :: corank, dim
           integer coindex
         end function
      end interface


!-------------------------------
!  coarray runtime interface
!-------------------------------
!! synchronization
      include "xmp_coarray_sync.h"
!!!      include "xmp_coarray_sync_sxace.h"

!! reference of coindexed objects
      include "xmp_coarray_get.h"

!! assignment statements to coindex variables
      include "xmp_coarray_put.h"

!! intrinsic subroutines atomic define/ref 
      include "xmp_coarray_atomic.h"

!! allocate statement
      include "xmp_coarray_alloc.h"

!! reduction subroutines
      include "xmp_coarray_reduction.h"

!! to reduce verbose messages from the native Fortran compiler
   external xmpf_coarray_prolog     !! F-BackEnd 
   external xmpf_coarray_epilog     !! F-BackEnd 

contains
      !-----------------------------------------------------------------
      !   inquery functions
      !-----------------------------------------------------------------
      function xmpf_coarray_uses_gasnet() result(answer)
        logical answer
#if _XMP_GASNET == 1
        answer = .true.
#else
        answer = .false.
#endif
        return
      end function

      function xmpf_coarray_uses_mpi3_onesided() result(answer)
        logical answer
#if _XMP_MPI3_ONESIDED == 1
        answer = .true.
#else
        answer = .false.
#endif
        return
      end function

      function xmpf_coarray_uses_fjrdma() result(answer)
        logical answer
#if _XMP_FJRDMA == 1
        answer = .true.
#else
        answer = .false.
#endif
        return
      end function

end module xmpf_coarray_decl

