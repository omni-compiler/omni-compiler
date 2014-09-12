program main

!$xmp nodes p2(2,2)
!$xmp template t2(16,19)
!$xmp distribute t2(block,block) onto p2

  real a2(16,16)
!$xmp align a2(i,j) with t2(i,j+3)

!$xmp loop (i,j) on t2(i,j+3)
  do j=1, 16
   do i=1, 16
     a2(i,j) = 0.
   enddo
  enddo

!$xmp task on p2(1,1)
  write(*,*) "PASS"
!$xmp end task

end program main
