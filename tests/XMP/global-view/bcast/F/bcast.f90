program test

  !$xmp nodes p(*)

  real :: c(100,100)
  real :: d(100,100)

  call sub(c, d)

  !$xmp task on p(1)
  write(*,*) "PASS"
  !$xmp end task

contains

  subroutine sub(c, d)

    real :: a(100,100)
    real, allocatable :: b(:,:)
    real :: c(100,100)
    real :: d(:,:)

    integer :: result = 0

    allocate (b(100,100))

    !$xmp task on p(1)
    a = 1.
    b = 2.
    c = 3.
    d = 4.
    !$xmp end task

! ---------------------------------------------

    !$xmp bcast (a)

    do j = 1, 100
       do i = 1, 100
          if (a(i,j) /= 1.) result = -1
       end do
    end do

    !$xmp reduction(+:result)

    !$xmp task on p(1)
    if (result /= 0) then
       write(*,*) "ERROR"
       call exit(1)
    endif
    !$xmp end task

! ---------------------------------------------

    !$xmp bcast (b)

    do j = 1, 100
       do i = 1, 100
          if (b(i,j) /= 2.) result = -1
       end do
    end do

    !$xmp reduction(+:result)

    !$xmp task on p(1)
    if (result /= 0) then
       write(*,*) "ERROR"
       call exit(1)
    endif
    !$xmp end task

! ---------------------------------------------

    !$xmp bcast (c)

    do j = 1, 100
       do i = 1, 100
          if (c(i,j) /= 3.) result = -1
       end do
    end do

    !$xmp reduction(+:result)

    !$xmp task on p(1)
    if (result /= 0) then
       write(*,*) "ERROR"
       call exit(1)
    endif
    !$xmp end task

! ---------------------------------------------

    !$xmp bcast (d)

    do j = 1, 100
       do i = 1, 100
          if (d(i,j) /= 4.) result = -1
       end do
    end do

    !$xmp reduction(+:result)

    !$xmp task on p(1)
    if (result /= 0) then
       write(*,*) "ERROR"
       call exit(1)
    endif
    !$xmp end task

  end program

end program test

