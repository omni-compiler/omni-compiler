program gmove
  implicit none
  type b
     real(8):: a(10)
  end type b
!$xmp nodes p(2)
!$xmp template t(10)
!$xmp distribute t(block) onto p
!$xmp align b%a(i) with t(i)
!$xmp align a(i) with t(i)
  type(b):: c
  real(8) :: a(10)
  integer :: i
  logical :: flag = .false.

!$xmp loop on t(i)
  do i=1, 10
     c%a(i) = i;
     a(i) = i + 100;
  end do
    
!$xmp gmove
  a(1:3) = c%a(6:8)

  if (xmp_node_num() == 1) then
    if (a(1) == 6 .and. a(2) == 7 .and. a(3) == 8 .and. a(4) == 104 .and. a(5) == 105) then
       flag = .true.
    end if
 else
    flag = .true.    
 end if
    
 !$xmp reduction (.and.:flag)
 if (flag .eqv. .true.) then
    if (xmp_node_num() == 1) then
       write(*,*) "OK"
    end if
 else
    if (xmp_node_num() == 1) then
       write(*,*) "Error!"
    end if
    call exit(1)
 end if
end program gmove

