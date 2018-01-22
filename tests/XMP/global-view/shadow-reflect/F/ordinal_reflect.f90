program test

!$xmp nodes p(2,2,2)

!$xmp template t(64,64,64)
!$xmp distribute t(block,block,block) onto p

  integer :: a(64,64,64), b(64,64,64), c(64,64,64)
!$xmp align (i,j,k) with t(i,j,k) :: a, b, c
!$xmp shadow (1,1,1) :: a, b, c

  integer :: result = 0

!$xmp loop (i,j,k) on t(i,j,k)
  do k = 1, 64
     do j = 1, 64
        do i = 1, 64
           a(i,j,k) = i * 10000 + j * 100 + k
           b(i,j,k) = i * 10000 + j * 100 + k
           c(i,j,k) = i * 10000 + j * 100 + k
        end do
     end do
  end do

!$xmp reflect (a) width(/periodic/1:1, 0, 0)
!$xmp reflect (a) width(0, /periodic/1:1, 0)
!$xmp reflect (a) width(0, 0, /periodic/1:1)

!$xmp reflect (b) width(/periodic/1:1, /periodic/1:1, /periodic/1:1)

!$xmp reflect (c) width(/periodic/1:1, /periodic/1:1, /periodic/1:1) async(100)
!$xmp wait_async(100)

!$xmp loop (i,j,k) on t(i,j,k) reduction(+:result)
  do k = 1, 64
  do j = 1, 64
  do i = 1, 64

     do kk = -1, 1
     do jj = -1, 1
     do ii = -1, 1

        if (a(i+ii, j+jj, k+kk) /= b(i+ii, j+jj, k+kk)) then
           result = -1
        end if

        if (a(i+ii, j+jj, k+kk) /= c(i+ii, j+jj, k+kk)) then
           result = -1
        end if

     end do
     end do
     end do

  end do
  end do
  end do

!$xmp task on p(1,1,1)
  if (result == 0) then
     write(*,*) "PASS"
  else
     write(*,*) "ERROR"
     call exit(1)
  endif
!$xmp end task

end program test
