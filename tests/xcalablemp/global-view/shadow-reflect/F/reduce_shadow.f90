program test

  !$xmp nodes p(4,4)

  integer m(4)
  integer n
  
  !$xmp template t(:,:)
  !$xmp distribute t(gblock(*),gblock(*)) onto p

  integer, allocatable :: a(:,:)
  !$xmp align a(i,j) with t(i,j)
  !$xmp shadow a(1:1,1:1)

  integer, allocatable :: b(:,:)

  integer :: result = 0

  n = 32
  m = (/4,4,8,16/)
  
  !$xmp template_fix(gblock(m),gblock(m)) t(n,n)
  allocate (a(n,n))

  allocate (b(n,n))
  
  !$xmp array on t(1:n,1:n)
  a(1:n,1:n) = 1

  do j = 1, n
     do i = 1, n
        b(i,j) = 1
        if (i == 1 .OR. i == 4 .OR. &
            i == 5 .OR. i == 8 .OR. &
            i == 9 .OR. i == 16 .OR. &
            i == 17 .OR. i == 32) then
           b(i,j) = b(i,j) + 1
        end if
        if (j == 1 .OR. j == 4 .OR. &
            j == 5 .OR. j == 8 .OR. &
            j == 9 .OR. j == 16 .OR. &
            j == 17 .OR. j == 32) then
           b(i,j) = b(i,j) + 1
        end if
     end do
  end do
  
  !$xmp reflect (a) width(/periodic/1,/periodic/1)

  !$xmp reduce_shadow (a) width(/periodic/1,/periodic/1) async(10)

  !$xmp wait_async (10)

  !$xmp loop (i,j) on t(i,j) reduction(+:result)
  do j = 1, n
     do i = 1, n
        if (a(i,j) /= b(i,j)) then
           result = 1
        end if
     end do
  end do

  !$xmp task on p(1,1)
  if (result /= 0) then
     write(*,*) "ERROR"
     call exit(1)
  else
     write(*,*) "PASS"
  endif
  !$xmp end task

end program test
