!! test of transfer function
!!   syntax:  transfer(source, mold [,size])
!!    If size is ommitted and mold is an array, the result is
!!    a one-dimensional array of the same type as mold, whose
!!    size is minimum and large enough to contain whole source.

  program transf

    type g1
       integer*4 a(3)
       real*4    r(2)
    end type g1

    integer len
    type(g1) :: ex2(2), goal2(2)
    character :: mold(0)

    do j=1,2
       ex2(j)%a = 7*j
       ex2(j)%r = 1.1*j
    enddo

    goal2 = transfer(transfer(ex2,mold),ex2)

    write(*,*) ex2
    write(*,*) goal2

  end program transf
