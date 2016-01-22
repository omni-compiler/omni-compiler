#!/bin/bash

#-------------------------------------------------------
#  generator for xmp_coarray_reduction.F90
#-------------------------------------------------------

TARGET="$1"

######################################
# procedure templates
######################################

template_broadcast='
subroutine xmpf_co_broadcast%dim%d_%tk%(source, source_image)
  include "mpif.h"
  %typeandkind%, intent(inout) :: source%shape%
  integer, intent(in) :: source_image
  integer ierr

  if (source_image <= 0) then
     call xmpf_coarray_fatal("The second argument of CO_BROADCAST must be positive.")
  end if
#ifdef _XMP_GASNET
  call mpi_barrier(mpi_comm_world, ierr, source, result)
  if (ierr /= 0) then
     call xmpf_coarray_fatal("CO_BROADCAST failed at mpi_barrier before mpi_bcast")
  end if
#endif
  call mpi_bcast(source, %size%, %mpitype%, source_image - 1, mpi_comm_world, ierr)
  if (ierr /= 0) then
     call xmpf_coarray_fatal("CO_BROADCAST failed in mpi_bcast")
  end if
#ifdef _XMP_GASNET
  call mpi_barrier(mpi_comm_world, ierr, source, result)
  if (ierr /= 0) then
     call xmpf_coarray_fatal("CO_BROADCAST failed at mpi_barrier after mpi_bcast")
  end if
#endif

  return
end subroutine'

template_reduction='
subroutine xmpf_co_%op%%dim%d_%tk%(source, result)
  include "mpif.h"
  %typeandkind%, intent(in) :: source%shape%
  %typeandkind%, intent(out) :: result%shape%
  integer ierr

#ifdef _XMP_GASNET
  call mpi_barrier(mpi_comm_world, ierr, source, result)
  if (ierr /= 0) then
     call xmpf_coarray_fatal("CO_%OP% failed at mpi_barrier before mpi_allreduce")
  end if
#endif
  call mpi_allreduce(source, result, %size%, %mpitype%, mpi_%op%, mpi_comm_world, ierr)
  if (ierr /= 0) then
     call xmpf_coarray_fatal("CO_%OP% failed in mpi_allreduce")
  end if
#ifdef _XMP_GASNET
  call mpi_barrier(mpi_comm_world, ierr, source, result)
  if (ierr /= 0) then
     call xmpf_coarray_fatal("CO_%OP% failed at mpi_barrier after mpi_allreduce")
  end if
#endif

  return
end subroutine'


######################################
# printer
######################################

print_broadcast() {
    dim="$4"
    case $dim in
        0) shape="";;
        1) shape="(:)";;
        2) shape="(:,:)";;
        3) shape="(:,:,:)";;
        4) shape="(:,:,:,:)";;
        5) shape="(:,:,:,:,:)";;
        6) shape="(:,:,:,:,:,:)";;
        7) shape="(:,:,:,:,:,:,:)";;
    esac
    case $dim in
        0) size="1";;
        *) size="size(source)";;
    esac

    echo "$template_broadcast" | sed '
        s/%typeandkind%/'$1'/g
        s/%tk%/'$2'/g
        s/%mpitype%/'$3'/g
        s/%dim%/'$dim'/g
        s/%shape%/'$shape'/g
        s/%size%/'$size'/g'
}

print_reduction() {
    dim="$6"
    case $dim in
        0) shape="";;
        1) shape="(:)";;
        2) shape="(:,:)";;
        3) shape="(:,:,:)";;
        4) shape="(:,:,:,:)";;
        5) shape="(:,:,:,:,:)";;
        6) shape="(:,:,:,:,:,:)";;
        7) shape="(:,:,:,:,:,:,:)";;
    esac
    case $dim in
        0) size="1";;
        *) size="size(source)";;
    esac

    echo "$template_reduction" | sed '
        s/%op%/'$1'/g
        s/%OP%/'$2'/g
        s/%typeandkind%/'$3'/g
        s/%tk%/'$4'/g
        s/%mpitype%/'$5'/g
        s/%dim%/'$dim'/g
        s/%shape%/'$shape'/g
        s/%size%/'$size'/g'
}


print_broadcast_all() {
    echo '
!-------------------------------
!  coarray intrinsic co_broadcast
!-------------------------------'

    for DIM in `seq 0 7`
    do
        if test "sxace-nec-superux" != "$TARGET"; then    ## SX-Fortran does not support integer(2).
            print_broadcast 'integer(2)' i2  mpi_integer2 ${DIM}
        fi
        print_broadcast 'integer(4)' i4  mpi_integer4 ${DIM}
        print_broadcast 'integer(8)' i8  mpi_integer8 ${DIM}

        print_broadcast 'real(4)'    r4  mpi_real4 ${DIM}
        print_broadcast 'real(8)'    r8  mpi_real8 ${DIM}

        print_broadcast 'complex(4)' z8  mpi_complex ${DIM}
        print_broadcast 'complex(8)' z16 mpi_double_complex ${DIM}

        if test "sxace-nec-superux" != "$TARGET"; then    ## SX-Fortran does not support logical(2).
            print_broadcast 'logical(2)' l2  mpi_integer2 ${DIM}
        fi
        print_broadcast 'logical(4)' l4  mpi_logical ${DIM}
    done
}


print_reduction_all() {
    echo '
!-------------------------------
!  coarray intrinsic co_'$1'
!-------------------------------'

    for DIM in `seq 0 7`
    do
        if test "sxace-nec-superux" != "$TARGET"; then    ## SX-Fortran does not support integer(2).
            print_reduction $1 $2 'integer(2)' i2  mpi_integer2 ${DIM}
        fi
        print_reduction $1 $2 'integer(4)' i4  mpi_integer4 ${DIM}
        print_reduction $1 $2 'integer(8)' i8  mpi_integer8 ${DIM}

        print_reduction $1 $2 'real(4)'    r4  mpi_real4 ${DIM}
        print_reduction $1 $2 'real(8)'    r8  mpi_real8 ${DIM}

        if test "$1" == "sum"; then
            print_reduction $1 $2 'complex(4)' z8  mpi_complex ${DIM}
            print_reduction $1 $2 'complex(8)' z16 mpi_double_complex ${DIM}
        fi
    done
}


######################################
# main
######################################

echo "!! This file is automatically generated by $0"
print_broadcast_all
print_reduction_all sum SUM
print_reduction_all max MAX
print_reduction_all min MIN

exit 0

