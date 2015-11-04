  program gmove_async

!$xmp nodes p(4)
!$xmp template t(8)
!$xmp distribute t(block) onto p

  integer a(8,8)
!$xmp align a(*,j) with t(j)

  integer b(8,8)
!$xmp align b(i,*) with t(i)

  integer :: ierr = 0

!$xmp loop (i) on t(i)
  do i = 1, 18
     do j = 1, 8
        b(i,j) = i * 10 + j
     end do
  end do

!$xmp gmove async(10)
  a(:,:) = b(:,:)

!$xmp wait_async(10)

!$xmp loop (j) on t(j) reduction(+:ierr)
  do j = 1, 8
     do i = 1, 8
        if (a(i,j) /= i * 10 + j) ierr = 1
     end do
  end do

!$xmp task on p(1)
  if (ierr == 0) then
     write(*,*) "PASS"
  else
     write(*,*) "ERROR"
     call exit(1)
  end if
!$xmp end task

end
