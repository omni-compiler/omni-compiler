module mmmm

!$xmp nodes p(2,2)
!$xmp template t(100,100)
!$xmp distribute t(block,block) onto p

contains

  subroutine sub1(x1, x2, y1, y2)
    real x1(100,100), x2(:,:), y1(100,100), y2(:,:)
!$xmp align (i,j) with t(i,j) :: x1, x2
!$xmp array on t(:,:)
    x1 = 1.0

!$xmp loop (i,j) on t(i,j)
    do j = 1, 100
       do i = 1, 100
          x2(i,j) = 1.0
       end do
    end do

    y1 = 1.0
    y2 = 1.0
  end subroutine sub1

  subroutine check(x1, x2, y1, y2, v)

    real x1(100,100), x2(:,:), y1(100,100), y2(:,:)
!$xmp align (i,j) with t(i,j) :: x1, x2
    real v

    integer :: result = 0

!$xmp loop (i,j) on t(i,j)
    do j = 1, 100
       do i = 1, 100
          if (x1(i,j) /= v) result = -1
       end do
    end do

!$xmp reduction(+:result)

!$xmp task on p(1,1)
    if ( result /= 0 ) then
       write(*,*) "ERROR"
       call exit(1)
       stop
    endif
!$xmp end task

!$xmp loop (i,j) on t(i,j)
    do j = 1, 100
       do i = 1, 100
          if (x2(i,j) /= v) result = -1
       end do
    end do

!$xmp reduction(+:result)

!$xmp task on p(1,1)
    if ( result /= 0 ) then
       write(*,*) "ERROR"
       call exit(1)
       stop
    endif
!$xmp end task

    do j = 1, 100
       do i = 1, 100
          if (y1(i,j) /= v) result = -1
       end do
    end do

!$xmp reduction(+:result)

!$xmp task on p(1,1)
    if ( result /= 0 ) then
       write(*,*) "ERROR"
       call exit(1)
       stop
    endif
!$xmp end task

    do j = 1, 100
       do i = 1, 100
          if (y2(i,j) /= v) result = -1
       end do
    end do

!$xmp reduction(+:result)

!$xmp task on p(1,1)
    if ( result /= 0 ) then
       write(*,*) "ERROR"
       call exit(1)
    endif
!$xmp end task

end subroutine check

end module mmmm

program test

  use mmmm

  interface
     subroutine sub3(x1, x2, y1, y2)
       real x1(100,100), x2(:,:), y1(100,100), y2(:,:)
     end subroutine sub3
  end interface

  real a1(100,100), a2(:,:), b1(100,100), b2(:,:)
  allocatable a2, b2
!$xmp align (i,j) with t(i,j) :: a1, a2

  allocate (a2(100,100), b2(100,100))

  call sub1(a1, a2, b1, b2)
  call check(a1, a2, b1, b2, 1.0)

  call sub2(a1, a2, b1, b2)
  call check(a1, a2, b1, b2, 2.0)

  call sub3(a1, a2, b1, b2)
  call check(a1, a2, b1, b2, 3.0)

!$xmp task on p(1,1)
  write(*,*) "PASS"
!$xmp end task

  contains

    subroutine sub2(x1, x2, y1, y2)
      real x1(100,100), x2(:,:), y1(100,100), y2(:,:)
!$xmp align (i,j) with t(i,j) :: x1, x2
!$xmp array on t(:,:)
      x1 = 2.0

!$xmp loop (i,j) on t(i,j)
      do j = 1, 100
         do i = 1, 100
            x2(i,j) = 2.0
         end do
      end do

      y1 = 2.0
      y2 = 2.0
    end subroutine sub2

end program test

subroutine sub3(x1, x2, y1, y2)
!$xmp nodes p(2,2)
!$xmp template t(100,100)
!$xmp distribute t(block,block) onto p
  real x1(100,100), x2(:,:), y1(100,100), y2(:,:)
!$xmp align (i,j) with t(i,j) :: x1, x2
!$xmp array on t(:,:)
  x1 = 3.0

!$xmp loop (i,j) on t(i,j)
  do j = 1, 100
     do i = 1, 100
        x2(i,j) = 3.0
     end do
  end do

  y1 = 3.0
  y2 = 3.0
end subroutine sub3
