program tasks

!$xmp nodes p(8)

!$xmp tasks

!$xmp task on p(1:4)
!$xmp barrier on p
!$xmp end task

!$xmp task on p(5:8)
!$xmp barrier on p
!$xmp end task

!$xmp end tasks

!$xmp task on p(1)
  write(*,*) "PASS"
!$xmp end task

end program tasks
