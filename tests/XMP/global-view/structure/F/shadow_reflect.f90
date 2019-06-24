program shadow_reflect
  implicit none
  type b
     real(8):: a(10)
  end type b
!$xmp nodes p(*)
!$xmp template t(10)
!$xmp distribute t(block) onto p
!$xmp align b%a(i) with t(i)
!$xmp shadow b%a(1)
  type(b):: c
  integer :: i
  logical :: flag = .false.

!$xmp loop on t(i)
  do i=1, 10
     c%a(i) = i;
  end do
!$xmp reflect (c%a)
  
  if (xmp_node_num() == 1) then
     if (c%a(6) == 6) then
        flag = .true.
     end if
  else
     if (c%a(5) == 5) then
        flag = .true.
     end if
  end if

!$xmp reduction(.and.:flag)
  
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
 
end program shadow_reflect
