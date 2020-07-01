program main
  include 'xmp_lib.h'
!$xmp nodes p(*)
!$xmp template t(3:12)
!$xmp distribute t(block) onto p
integer a(10)
!$xmp align a(i) with t(i+2)

!$xmp loop on t(i+2)
do i = 3, 10
   a(i) = i
end do

!$xmp task on p(1)
  write(*,*) "PASS"
!$xmp end task
  
end program main
