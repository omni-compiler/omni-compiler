program test

  !$xmp nodes p(2,2)
  !$xmp template t(8,8)
  !$xmp distribute t(block,block) onto p

  integer, pointer :: a(:,:,:)
  !$xmp align a(*,j,k) with t(j,k)
  !$xmp shadow a(0,1,1)

  integer, target :: b(8,8,8)
  !$xmp align b(*,j,k) with t(j,k)
  !$xmp shadow b(0,1,1)

  integer c(8,8,8)
  !$xmp align c(*,j,k) with t(j,k)
  !$xmp shadow c(0,1,1)

  integer result = 0

  !$xmp loop on t(j,k)
  do k = 1, 8
  do j = 1, 8
  do i = 1, 8
     b(i,j,k) = i * 100 + j * 10 + k
     c(i,j,k) = i * 100 + j * 10 + k
  end do
  end do
  end do
  
  a => b

  !$xmp reflect (a) width(0, /periodic/1:1, /periodic/1:1)
  !$xmp reflect (c) width(0, /periodic/1:1, /periodic/1:1)

  !$xmp loop on t(j,k) reduction(+:result)
  do k = 1, 8
  do j = 1, 8
  do i = 1, 8
     if (a(i,j,k) /= c(i,j,k)) result = 1
  end do
  end do
  end do

  !$xmp loop on t(j,k) reduction(+:result)
  do k = 1, 8
  do j = 1, 8
  do i = 1, 8

     do kk = -1, 1
     do jj = -1, 1

        if (a(i, j+jj, k+kk) /= c(i, j+jj, k+kk)) then
           result = -1
        end if

     end do
     end do

  end do
  end do
  end do

  !$xmp task on p(1,1)
  if (result /= 0)  then
     write(*,*) "ERROR"
     call exit(1)
  endif

  write(*,*) "PASS"
  !$xmp end task
  
end program test
