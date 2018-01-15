!! Ver3: NG --- Error: There is no specific subroutine for the generic 'xmpf_coarray_malloc_generic'
!! Ver4:

  program alloc1

    type g0
       integer*8 a(10)
       real*8    r(10)
    end type g0

    type g1
       integer*8 a(10)
       real*8    r(10)
       type(g0),pointer :: p(:)
    end type g1

    type(g1), save :: s1, cs1[*]
    type(g1), save :: a1(3), ca1(3)[*]

    type(g0), allocatable :: p1(:,:), cp1(:,:)[:,:]
    character(3), allocatable :: c1(:), cc1(:)[:]
    
    integer isize(3), csize(3)

    !!---- test #1
    isize(1) = sizeof(s1) + sizeof(a1)
    csize(1) = xmpf_coarray_allocated_bytes()

    write(*,*) "isize(1) = ", isize(1)
    write(*,*) "csize(1) = ", csize(1), " should be >= isize(1)."

    !!---- test #2
    allocate (p1(2,3))
    allocate (cp1(2,3)[2,*])

    isize(2) = (80+80)*2*3
    csize(2) = xmpf_coarray_allocated_bytes() - csize(1)

    write(*,*) "isize(2) = ", isize(2)
    write(*,*) "csize(2) = ", csize(2), " should be >= isize(2)."

    !!---- test #3
!!    allocate (c1(4))
!!    allocate (cc1(4)[*])

!!    isize(3) = 12
!!    csize(3) = xmpf_coarray_allocated_bytes() - csize(1) - csize(2)

!!    write(*,*) "isize(3) = ", isize(3)
!!    write(*,*) "csize(3) = ", csize(3), " should be >= isize(3)."

    !!---- check
    nerr=0
    me = this_image()
!!    do i=1,3
    do i=1,2
       if (csize(i)<isize(i)) then
          nerr=nerr+1
          write(*,101) me, csize(i), isize(i)
       endif
    enddo

101 format("[",i0,"] allocated size (",i0,") is smaller than the data size (",i0,").")

    if (nerr==0) then 
       print '("[",i0,"] OK")', me
    else
       print '("[",i0,"] number of NGs: ",i0)', me, nerr
    end if

  end program alloc1
