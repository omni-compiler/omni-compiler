program	loop
  implicit none
  type b
     real(8):: a(10)
  end type b
!$xmp nodes p(*)
!$xmp template t(10)
!$xmp distribute t(block) onto p
!$xmp align b%a(i) with t(i)
  type(b):: c
  integer :: i, sum = 0

!$xmp loop on t(i)
  do i=1, 10
     c%a(i) = i
  end do

!$xmp loop on t(i) reduction(+:sum)
  do i=1, 10
     sum = sum + c%a(i)
  end do
  
  if (sum == 55) then
     if (xmp_node_num() == 1) then
        write(*,*) "OK"
     end if
  else
     if (xmp_node_num() == 1) then
        write(*,*) "Error!"
     end if
     call exit(1)
  end if
end program loop

  
