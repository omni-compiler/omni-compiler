!! XMP/F Coarray Declarations 
!!
!! This file is use-associated to the user program via the USE
!! statement that is inserted automatically.
!!
!! In addition, this file is referred from a shellscript in backend:
!!   F-BackEnd/src/xcodeml/f/decompile/
!!     XfDecompileDomVisitor_coarrayLibs.java.sh
!!

module xmpf_coarray_decl

  !-------------------------------
  !  coarray intrinsics
  !-------------------------------

  interface
     integer function xmpf_image_index(descptr, cosubs)
       integer(8), intent(in) :: descptr
       integer, intent(in) :: cosubs(*)
     end function xmpf_image_index
  end interface

  interface xmpf_image_index_generic
     module procedure xmpf_image_index_dim1
     module procedure xmpf_image_index_dim2
     module procedure xmpf_image_index_dim3
     module procedure xmpf_image_index_dim4
     module procedure xmpf_image_index_dim5
     module procedure xmpf_image_index_dim6
     module procedure xmpf_image_index_dim7
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
     end function xmpf_num_images_current
  end interface

  interface xmpf_this_image_generic
     function xmpf_this_image_current() result(image)
       integer image
     end function xmpf_this_image_current
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

  !! reference of coindexed objects
  include "xmp_coarray_get.h"
  include "xmp_coarray_getsub.h"

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
  integer function xmpf_image_index_dim1(descptr, &
       cs1) result(index)
    integer :: index
    integer(8), intent(in) :: descptr
    integer, intent(in) :: cs1
    index = xmpf_image_index &
         (descptr, (/cs1/))
  end function xmpf_image_index_dim1

  integer function xmpf_image_index_dim2(descptr, &
       cs1, cs2) result(index)
    integer :: index
    integer(8), intent(in) :: descptr
    integer, intent(in) :: cs1, cs2
    index = xmpf_image_index &
         (descptr, (/cs1, cs2/))
  end function xmpf_image_index_dim2

  integer function xmpf_image_index_dim3(descptr, &
       cs1, cs2, cs3) result(index)
    integer :: index
    integer(8), intent(in) :: descptr
    integer, intent(in) :: cs1, cs2, cs3
    index = xmpf_image_index &
         (descptr, (/cs1, cs2, cs3/))
  end function xmpf_image_index_dim3

  integer function xmpf_image_index_dim4(descptr, &
       cs1, cs2, cs3, cs4) result(index)
    integer :: index
    integer(8), intent(in) :: descptr
    integer, intent(in) :: cs1, cs2, cs3, cs4
    index = xmpf_image_index &
         (descptr, (/cs1, cs2, cs3, cs4/))
  end function xmpf_image_index_dim4

  integer function xmpf_image_index_dim5(descptr, &
       cs1, cs2, cs3, cs4, cs5) result(index)
    integer :: index
    integer(8), intent(in) :: descptr
    integer, intent(in) :: cs1, cs2, cs3, cs4, cs5
    index = xmpf_image_index &
         (descptr, (/cs1, cs2, cs3, cs4, cs5/))
  end function xmpf_image_index_dim5

  integer function xmpf_image_index_dim6(descptr, &
       cs1, cs2, cs3, cs4, cs5, cs6) result(index)
    integer :: index
    integer(8), intent(in) :: descptr
    integer, intent(in) :: cs1, cs2, cs3, cs4, cs5, cs6
    index = xmpf_image_index &
         (descptr, (/cs1, cs2, cs3, cs4, cs5, cs6/))
  end function xmpf_image_index_dim6

  integer function xmpf_image_index_dim7(descptr, &
       cs1, cs2, cs3, cs4, cs5, cs6, cs7) result(index)
    integer :: index
    integer(8), intent(in) :: descptr
    integer, intent(in) :: cs1, cs2, cs3, cs4, cs5, cs6, cs7
    index = xmpf_image_index &
         (descptr, (/cs1, cs2, cs3, cs4, cs5, cs6, cs7/))
  end function xmpf_image_index_dim7


  function xmpf_coarray_uses_gasnet() result(answer)
    logical answer
#if _XMP_GASNET == 1
    answer = .true.
#else
    answer = .false.
#endif
    return
  end function xmpf_coarray_uses_gasnet

  function xmpf_coarray_uses_mpi3_onesided() result(answer)
    logical answer
#if _XMP_MPI3_ONESIDED == 1
    answer = .true.
#else
    answer = .false.
#endif
    return
  end function xmpf_coarray_uses_mpi3_onesided

  function xmpf_coarray_uses_fjrdma() result(answer)
    logical answer
#if _XMP_FJRDMA == 1
    answer = .true.
#else
    answer = .false.
#endif
    return
  end function xmpf_coarray_uses_fjrdma

end module xmpf_coarray_decl

