!! test of transfer function
!!   syntax:  transfer(source, mold [,size])
!!    If size is ommitted and mold is an array, the result is
!!    a one-dimensional array of the same type as mold, whose
!!    size is minimum and large enough to contain whole source.

  program transf

    type g1
       integer*4 a(3)
       real*8    r(3)
    end type g1

    integer len
    type(g1) :: ex, goal
    character :: mold(0)

    ex%a = 3
    ex%r = 1.2

    goal = transfer(transfer(ex,mold),ex)

    write(*,*) goal
  end program transf
