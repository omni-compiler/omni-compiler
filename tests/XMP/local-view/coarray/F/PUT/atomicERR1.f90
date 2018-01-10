  program main

    integer ii[*], iiok, nn, nnok
    logical la(3,2)[*], laok, lb
    real x[*]

    ii=3
    nn=0
    la=.false.
    me=this_image()
    sync all

    nerr=0

!--- type (error in native compiler)
!!    call atomic_define(x, 1)        !! no specific subroutine

!--- noncoarray
!!    call atomic_define(iiok, 1)
!!    call atomic_define(lb, .false.)

!--- illegal expression
!!    call atomic_define(ii+1, -1)
!!    call atomic_define((ii), -1)      !! This is not detected in FE

!--- not scalar (error in native compiler)
!!    call atomic_define(la(1:2,1), .true.)        !! no specific subroutine

    end
